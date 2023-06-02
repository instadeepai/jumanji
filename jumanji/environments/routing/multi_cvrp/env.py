# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib
from numpy.typing import NDArray

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.multi_cvrp.generator import (
    Generator,
    UniformRandomGenerator,
)
from jumanji.environments.routing.multi_cvrp.reward import DenseReward, RewardFn
from jumanji.environments.routing.multi_cvrp.types import (
    Node,
    Observation,
    ObsVehicle,
    PenalityCoeff,
    State,
    StateVehicle,
    TimeWindow,
)
from jumanji.environments.routing.multi_cvrp.utils import (
    DEPOT_IDX,
    compute_distance,
    compute_time_penalties,
    create_action_mask,
    max_single_vehicle_distance,
)
from jumanji.environments.routing.multi_cvrp.viewer import MultiCVRPViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class MultiCVRP(Environment[State]):
    """
    Multi-Vehicle Routing Problems with Soft Time Windows (MVRPSTW) environment as described in [1].
    We simplfy the naming to multi-agent capacitated vehicle routing problem (MultiCVRP).

    - reward: jax array (float32)
        this global reward is provided to each agent. The reward is equal to the negative sum of the
        distances between consecutive nodes at the end of the episode over all agents. All time
        penalties are also added to the reward.

    - observation and state:
        the observation and state variable types are defined in:
        jumanji/environments/routing/multi_cvrp/types.py

    [1] Zhang et al. (2020). "Multi-Vehicle Routing Problems with Soft Time Windows: A
    Multi-Agent Reinforcement Learning Approach".
    """

    def __init__(
        self,
        generator: Optional[Generator] = None,
        reward_fn: Optional[RewardFn] = None,
        viewer: Optional[Viewer] = None,
    ):
        """
        Instantiates a `MultiCVRP` environment.

        Args:
            generator: `Generator` whose `__call__` instantiates an environment instance.
                Implemented options are [`UniformRandomGenerator`].
                Defaults to `UniformRandomGenerator` with `num_customers=20` and `num_vehicles=2`.
            reward_fn: `RewardFn` whose `__call__` method computes the reward of an environment
                transition. The function must compute the reward based on the current state
                and whether the environment is done.
                Implemented options are [`DenseReward`, `SparseReward`]. Defaults to `DenseReward`.
            viewer: `Viewer` used for rendering. Defaults to `MultiCVRPViewer` with "human" render
                mode.
        """

        # Create generator used for generating new environments
        self._generator = generator or UniformRandomGenerator(
            num_customers=20,
            num_vehicles=2,
        )

        self._max_capacity = self._generator._max_capacity
        self._map_max = self._generator._map_max
        self._customer_demand_max = self._generator._customer_demand_max
        self._max_start_window = self._generator._max_start_window
        self._max_end_window = self._generator._max_end_window
        self._time_window_length = self._generator._time_window_length
        self._early_coef_rand = self._generator._early_coef_rand
        self._late_coef_rand = self._generator._late_coef_rand
        self._num_customers = self._generator._num_customers
        self._num_vehicles = self._generator._num_vehicles

        # Create reward function used for computing rewards
        self._reward_fn = reward_fn or DenseReward(
            self._num_vehicles, self._num_customers, self._map_max
        )

        # Create viewer used for rendering
        self._viewer = viewer or MultiCVRPViewer(
            name="MultiCVRP",
            num_vehicles=self._num_vehicles,
            num_customers=self._num_customers,
            map_max=self._map_max,
            render_mode="human",
        )

        # From the paper: All distances are represented by Euclidean distances in the plane,
        # and the speeds of all  vehicles are assumed to be identical (i.e., it takes one
        # unit of time to travel one unit of distance)
        self._speed: int = 1

        self._max_local_time = (
            max_single_vehicle_distance(self._map_max, self._num_customers)
            / self._speed
        )

    def __repr__(self) -> str:
        return f"MultiCVRP(num_customers={self._num_customers}, num_vehicles={self._num_vehicles})"

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """
        Resets the environment.

        Args:
            key: used to randomly generate the problem and the start node.

        Returns:
             state: State object corresponding to the new state of the environment.
             timestep: TimeStep object corresponding to the first timestep returned by the
                environment.
        """

        state = self._generator(key)

        timestep = restart(observation=self._state_to_observation(state))

        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        """
        Run one timestep of the environment's dynamics.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the index of the next nodes to visit.

        Returns:
            state, timestep: Tuple[State, TimeStep] containing the next state of the environment,
                as well as the timestep to be observed.
        """
        new_state = self._update_state(state, action)

        is_done = (
            (new_state.nodes.demands.sum() == 0)
            & (new_state.vehicles.positions == DEPOT_IDX).all()
        ) | jnp.any(new_state.step_count > self._num_customers * 2)

        reward = self._reward_fn(state, new_state, is_done)

        timestep = self._state_to_timestep(new_state, reward, is_done)

        return new_state, timestep

    def observation_spec(self) -> specs.Spec[Observation]:
        """
        Returns the observation spec.

        Returns:
            observation_spec: a Tuple containing the spec for each of the constituent fields of an
                observation.
        """

        node_coordinates = specs.BoundedArray(
            shape=(self._num_customers + 1, 2),
            minimum=0.0,
            maximum=self._map_max,
            dtype=jnp.float32,
            name="node_coordinates",
        )

        node_demands = specs.BoundedArray(
            shape=(self._num_customers + 1,),
            minimum=0,
            maximum=self._max_capacity,
            dtype=jnp.int16,
            name="node_demands",
        )

        node_time_windows_start = specs.BoundedArray(
            shape=(self._num_customers + 1,),
            minimum=0.0,
            maximum=self._max_end_window,
            dtype=jnp.float32,
            name="node_time_windows_start",
        )

        node_time_windows_end = specs.BoundedArray(
            shape=(self._num_customers + 1,),
            minimum=0.0,
            maximum=self._max_end_window,
            dtype=jnp.float32,
            name="node_time_windows_end",
        )

        node_penalty_coeffs_start = specs.BoundedArray(
            shape=(self._num_customers + 1,),
            minimum=0.0,
            maximum=self._late_coef_rand[-1],
            dtype=jnp.float32,
            name="node_penalty_coeffs_start",
        )

        node_penalty_coeffs_end = specs.BoundedArray(
            shape=(self._num_customers + 1,),
            minimum=0.0,
            maximum=self._late_coef_rand[-1],
            dtype=jnp.float32,
            name="node_penalty_coeffs_end",
        )

        vehicle_coordinates = specs.BoundedArray(
            shape=(self._num_vehicles, 2),
            minimum=0.0,
            maximum=self._map_max,
            dtype=jnp.float32,
            name="vehicle_coordinates",
        )

        vehicle_local_time = specs.BoundedArray(
            shape=(self._num_vehicles,),
            minimum=0.0,
            maximum=self._max_local_time,
            dtype=jnp.float32,
            name="vehicle_local_time",
        )

        vehicle_capacity = specs.BoundedArray(
            shape=(self._num_vehicles,),
            minimum=0,
            maximum=self._max_capacity,
            dtype=jnp.int16,
            name="vehicle_capacity",
        )

        action_mask = specs.BoundedArray(
            shape=(self._num_vehicles, self._num_customers + 1),
            minimum=0,
            maximum=self._num_customers + 1,
            dtype=jnp.bool_,
            name="action_mask",
        )

        # Node spec
        nodes_spec = specs.Spec(
            Node, "NodesSpec", coordinates=node_coordinates, demands=node_demands
        )

        # Window spec
        windows_spec = specs.Spec(
            TimeWindow,
            "WindowSpec",
            start=node_time_windows_start,
            end=node_time_windows_end,
        )

        # Penality spec
        penality_spec = specs.Spec(
            PenalityCoeff,
            "PenalitySpec",
            early=node_penalty_coeffs_start,
            late=node_penalty_coeffs_end,
        )

        # Main vehicle spec
        vehicle_spec = specs.Spec(
            ObsVehicle,
            "MainVehicleSpec",
            coordinates=vehicle_coordinates,
            local_times=vehicle_local_time,
            capacities=vehicle_capacity,
        )

        return specs.Spec(
            Observation,
            "ObservationSpec",
            nodes=nodes_spec,
            windows=windows_spec,
            coeffs=penality_spec,
            vehicles=vehicle_spec,
            action_mask=action_mask,
        )

    def action_spec(self) -> specs.BoundedArray:
        """
        Returns the action spec.

        Returns:
            action_spec: a `specs.BoundedArray` spec.
        """

        return specs.BoundedArray(
            (self._num_vehicles,),
            jnp.int16,
            0,
            self._num_customers + 1,
            "actions",
        )

    def render(
        self,
        state: State,
    ) -> Optional[NDArray]:
        """Render the given state of the environment. This rendering shows the layout of the tour so
            far with the cities as circles, and the depot as a square.

        Args:
            state: environment state to render.
            save_path: the optional path where the image should be saved.

        Returns:
            rgb_array: the RGB image of the state as an array.
        """
        return self._viewer.render(state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of states.

        Args:
            states: sequence of `State` corresponding to subsequent timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            animation that can export to gif, mp4, or render with HTML.
        """
        return self._viewer.animate(states, interval, save_path)

    def _update_state(self, state: State, action: chex.Array) -> State:
        """
        Updates the state of the environment.

        Args:
            state: State object containing the dynamics of the environment.
            action: int, index of the next node to visit.

        Returns:
            state: State object corresponding to the new state of the environment.
        """

        # Convert the action to int16.
        next_nodes = jnp.int16(action)

        # Zero any node selections if the node has zero demand or does not have enough
        # capacity left. If vehicles chose to go to the depot the demand criteria
        # will not be met which sends them back to the depot.
        next_nodes = (
            next_nodes
            * (state.vehicles.capacities >= state.nodes.demands[next_nodes])
            * (state.nodes.demands[next_nodes] > 0)
        )

        # Zero node selections where more than one vehicle selected a valid conditional
        # action to visit the same node.
        values, unique_indices = jnp.unique(
            next_nodes, return_index=True, size=self._num_vehicles
        )
        next_nodes = jnp.zeros(len(next_nodes), dtype=next_nodes.dtype)
        next_nodes = next_nodes.at[unique_indices].set(values)

        # Update the vehicle distances and local times.
        start_coords = state.nodes.coordinates[state.vehicles.positions]
        end_coords = state.nodes.coordinates[next_nodes]
        step_travel_distances = compute_distance(start_coords, end_coords)
        vehicle_distances = state.vehicles.distances + step_travel_distances
        vehicle_local_times = (
            state.vehicles.local_times + step_travel_distances / self._speed
        )

        # Update the vehicle time penalties.
        vehicle_time_penalties = state.vehicles.time_penalties + compute_time_penalties(
            vehicle_local_times,
            state.windows.start[next_nodes],
            state.windows.end[next_nodes],
            state.coeffs.early[next_nodes],
            state.coeffs.late[next_nodes],
        )

        # Calculate the new vehicle capacities. Restore vehicle capacities at depot.
        vehicle_capacities = jnp.where(
            next_nodes == DEPOT_IDX,
            self._max_capacity,
            state.vehicles.capacities - state.nodes.demands[next_nodes],
        )

        # Calculate the new node demands.
        node_demands = state.nodes.demands.at[next_nodes].set(0)

        # Update order (used for rendering)
        order = state.order.at[:, state.step_count].set(next_nodes)

        return State(
            nodes=Node(
                coordinates=state.nodes.coordinates,
                demands=node_demands,
            ),
            windows=state.windows,
            coeffs=state.coeffs,
            vehicles=StateVehicle(
                local_times=vehicle_local_times,
                positions=next_nodes,
                capacities=vehicle_capacities,
                distances=vehicle_distances,
                time_penalties=vehicle_time_penalties,
            ),
            step_count=state.step_count + 1,
            order=order,
            action_mask=create_action_mask(node_demands, vehicle_capacities),
            key=state.key,
        )

    def _state_to_observation(self, state: State) -> Observation:
        """
        Converts a state into an observation.

        Args:
            state: State object containing the dynamics of the environment.

        Returns:
            observation: Observation object containing the observation of the environment.
        """

        vehicle_coordinates = state.nodes.coordinates[state.vehicles.positions]

        return Observation(
            nodes=Node(
                coordinates=state.nodes.coordinates,
                demands=state.nodes.demands,
            ),
            windows=TimeWindow(
                start=state.windows.start,
                end=state.windows.end,
            ),
            coeffs=PenalityCoeff(
                early=state.coeffs.early,
                late=state.coeffs.late,
            ),
            vehicles=ObsVehicle(
                coordinates=vehicle_coordinates,
                local_times=state.vehicles.local_times,
                capacities=state.vehicles.capacities,
            ),
            action_mask=state.action_mask,
        )

    def _state_to_timestep(
        self, state: State, reward: chex.Numeric, is_done: bool
    ) -> TimeStep:
        """
        Checks if the state is terminal and converts it into a timestep.

        Args:
            state: State object containing the dynamics of the environment.

        Returns:
            timestep: TimeStep object containing the timestep of the environment.
        """

        observation = self._state_to_observation(state)

        timestep: TimeStep = jax.lax.cond(
            is_done,
            termination,
            transition,
            reward,
            observation,
        )
        return timestep
