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
import matplotlib
from numpy.typing import NDArray

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.macvrp.generator import (
    Generator,
    UniformRandomGenerator,
)
from jumanji.environments.routing.macvrp.reward import RewardFn, SparseReward
from jumanji.environments.routing.macvrp.types import (
    Node,
    Observation,
    ObsVehicle,
    PenalityCoeff,
    State,
    StateVehicle,
    TimeWindow,
)
from jumanji.environments.routing.macvrp.utils import (
    DEPOT_IDX,
    compute_distance,
    compute_time_penalties,
    max_single_vehicle_distance,
)
from jumanji.environments.routing.macvrp.viewer import MACVRPViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class MACVRP(Environment[State]):
    """
    Multi-Vehicle Routing Problems with Soft Time Windows (MVRPSTW) environment as described in [1].
    We simplfy the naming to multi-agent capacitated vehicle routing problem (MACVRP).

    - reward: jax array (float32)
        this global reward is provided to each agent. The reward is equal to the negative sum of the
        distances between consecutive nodes at the end of the episode over all agents. All time
        penalties are also added to the reward.

    - observation and state:
        the observation and state variable types are defined in:
        jumanji/environments/routing/macvrp/types.py

    [1] Zhang et al. (2020). "Multi-Vehicle Routing Problems with Soft Time Windows: A
    Multi-Agent Reinforcement Learning Approach".
    """

    """Instantiates a `MACVRP` environment.

        Args:
            num_nodes: number of city nodes in the environment. Defaults to 20.
            max_capacity: maximum capacity of the vehicle. Defaults to 30.
            max_demand: maximum demand of each node. Defaults to 10.
            reward_fn: `RewardFn` whose `__call__` method computes the reward of an environment
                transition. The function must compute the reward based on the current state,
                the chosen action, the next state and whether the action is valid.
                Implemented options are [`SparseReward`]. Defaults to `DenseReward`.
            viewer: `Viewer` used for rendering. Defaults to `CVRPViewer` with "human" render mode.
        """

    def __init__(
        self,
        generator: Optional[Generator] = None,
        reward_fn: Optional[RewardFn] = None,
        viewer: Optional[Viewer] = None,
    ):
        """
        Instantiates a `MACVRP` environment.

        Args:
            generator: `Generator` whose `__call__` instantiates an environment instance.
                Implemented options are [`UniformRandomGenerator`].
                Defaults to `UniformRandomGenerator` with `num_customers=6` and `num_vehicles=2`.
            reward_fn: `RewardFn` whose `__call__` method computes the reward of an environment
                transition. The function must compute the reward based on the current state
                and whether the environment is done.
                Implemented options are [`DenseReward`]. Defaults to `DenseReward`.
            viewer: `Viewer` used for rendering. Defaults to `MACVRPViewer` with "human" render
                mode.
        """

        # Create generator used for generating new environments
        self._generator = generator or UniformRandomGenerator(
            num_customers=6,
            num_vehicles=2,
        )

        self._max_capacity = self._generator._max_capacity
        self._map_max = self._generator._map_max
        self._customer_demand_max = self._generator._customer_demand_max
        self._max_start_window = self._generator._max_start_window
        self._max_end_window = self._generator._max_end_window
        self._time_window_length = self._generator._time_window_length
        self._late_coef_rand = self._generator._late_coef_rand
        self._num_customers = self._generator._num_customers
        self._num_vehicles = self._generator._num_vehicles

        # Create reward function used for computing rewards
        self._reward_fn = reward_fn or SparseReward(
            self._num_vehicles, self._num_customers, self._map_max
        )

        # Create viewer used for rendering
        self._viewer = viewer or MACVRPViewer(
            name="MACVRP",
            num_vehicles=self._num_vehicles,
            num_customers=self._num_customers,
            map_max=self._map_max,
            render_mode="human",
        )

        self._max_local_time = (
            max_single_vehicle_distance(self._map_max, self._num_customers)
            / self._speed
        )

        # From the paper: All distances are represented by Euclidean distances in the plane,
        # and the speeds of all  vehicles are assumed to be identical (i.e., it takes one
        # unit of time to travel one unit of distance)
        self._speed: int = 1

    def __repr__(self) -> str:
        return f"MACVRP(num_customers={self._num_customers}, num_vehicles={self._num_vehicles})"

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
        self, state: State, actions: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        """
        Run one timestep of the environment's dynamics.

        Args:
            state: State object containing the dynamics of the environment.
            actions: Array containing the index of the next nodes to visit.

        Returns:
            state, timestep: Tuple[State, TimeStep] containing the next state of the environment,
                as well as the timestep to be observed.
        """
        state = self._update_state(state, actions)

        timestep = self._state_to_timestep(state)
        return state, timestep

    def observation_spec(self) -> specs.Spec[Observation]:
        """
        Returns the observation spec.

        Returns:
            observation_spec: a Tuple containing the spec for each of the constituent fields of an
                observation.
        """

        node_coordinates = specs.BoundedArray(
            shape=(self._num_vehicles, self._num_customers + 1, 2),
            minimum=0.0,
            maximum=self._map_max,
            dtype=jax.numpy.float32,
            name="node_coordinates",
        )

        node_demands = specs.BoundedArray(
            shape=(
                self._num_vehicles,
                self._num_customers + 1,
            ),
            minimum=0,
            maximum=self._max_capacity,
            dtype=jax.numpy.int16,
            name="node_demands",
        )

        node_time_windows_start = specs.BoundedArray(
            shape=(self._num_vehicles, self._num_customers + 1),
            minimum=0.0,
            maximum=self._max_end_window,
            dtype=jax.numpy.float32,
            name="node_time_windows_start",
        )

        node_time_windows_end = specs.BoundedArray(
            shape=(self._num_vehicles, self._num_customers + 1),
            minimum=0.0,
            maximum=self._max_end_window,
            dtype=jax.numpy.float32,
            name="node_time_windows_end",
        )

        node_penalty_coeffs_start = specs.BoundedArray(
            shape=(self._num_vehicles, self._num_customers + 1),
            minimum=0.0,
            maximum=self._late_coef_rand[-1],
            dtype=jax.numpy.float32,
            name="node_penalty_coeffs_start",
        )

        node_penalty_coeffs_end = specs.BoundedArray(
            shape=(self._num_vehicles, self._num_customers + 1),
            minimum=0.0,
            maximum=self._late_coef_rand[-1],
            dtype=jax.numpy.float32,
            name="node_penalty_coeffs_end",
        )

        other_vehicles_positions = specs.BoundedArray(
            shape=(self._num_vehicles, self._num_vehicles - 1),
            minimum=0,
            maximum=self._num_customers + 1,
            dtype=jax.numpy.int16,
            name="other_vehicles_positions",
        )

        other_vehicles_local_times = specs.BoundedArray(
            shape=(
                self._num_vehicles,
                self._num_vehicles - 1,
            ),
            minimum=0.0,
            maximum=self._max_local_time,
            dtype=jax.numpy.float32,
            name="other_vehicles_local_times",
        )

        other_vehicles_capacities = specs.BoundedArray(
            shape=(
                self._num_vehicles,
                self._num_vehicles - 1,
            ),
            minimum=0,
            maximum=self._max_capacity,
            dtype=jax.numpy.int16,
            name="other_vehicles_capacities",
        )

        vehicle_position = specs.BoundedArray(
            shape=(self._num_vehicles,),
            minimum=0,
            maximum=self._num_customers + 1,
            dtype=jax.numpy.int16,
            name="vehicle_position",
        )

        vehicle_local_time = specs.BoundedArray(
            shape=(self._num_vehicles,),
            minimum=0.0,
            maximum=self._max_local_time,
            dtype=jax.numpy.float32,
            name="vehicle_local_time",
        )

        vehicle_capacity = specs.BoundedArray(
            shape=(self._num_vehicles,),
            minimum=0,
            maximum=self._max_capacity,
            dtype=jax.numpy.int16,
            name="vehicle_capacity",
        )

        action_mask = specs.BoundedArray(
            shape=(self._num_vehicles, self._num_customers + 1),
            minimum=0,
            maximum=self._num_customers + 1,
            dtype=jax.numpy.bool_,
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

        # Other vehicle spec
        other_vehicle_spec = specs.Spec(
            ObsVehicle,
            "OtherVehicleSpec",
            positions=other_vehicles_positions,
            local_times=other_vehicles_local_times,
            capacities=other_vehicles_capacities,
        )

        # Main vehicle spec
        main_vehicle_spec = specs.Spec(
            ObsVehicle,
            "MainVehicleSpec",
            positions=vehicle_position,
            local_times=vehicle_local_time,
            capacities=vehicle_capacity,
        )

        return specs.Spec(
            Observation,
            "ObservationSpec",
            nodes=nodes_spec,
            windows=windows_spec,
            coeffs=penality_spec,
            other_vehicles=other_vehicle_spec,
            main_vehicles=main_vehicle_spec,
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
            jax.numpy.int16,
            0,
            self._num_customers + 1,
            "actions",
        )

    def render(self, state: State) -> Optional[NDArray]:
        """Render the given state of the environment. This rendering shows the layout of the tour so
            far with the cities as circles, and the depot as a square.

        Args:
            state: environment state to render.

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

    def _update_state(self, state: State, actions: chex.Array) -> State:
        """
        Updates the state of the environment.

        Args:
            state: State object containing the dynamics of the environment.
            next_node: int, index of the next node to visit.

        Returns:
            state: State object corresponding to the new state of the environment.
        """

        # Convert actions to int16.
        next_nodes = jax.numpy.int16(actions)

        # Zero any node selections if the node has zero demand or does not have enough
        # capacity left.
        next_nodes = (
            next_nodes
            * (state.vehicles.capacities > state.nodes.demands[next_nodes])
            * (state.nodes.demands[next_nodes] > 0)
        )

        # Zero node selections where more than one vehicle selected a valid conditional
        # action to visit the same node.
        values, unique_indices = jax.numpy.unique(
            next_nodes, return_index=True, size=self._num_vehicles
        )

        next_nodes = jax.numpy.zeros(len(next_nodes), dtype=next_nodes.dtype)
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
        vehicle_capacities = jax.numpy.where(
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

        other_vehicles_positions = jax.numpy.zeros(
            (self._num_vehicles, self._num_vehicles - 1), dtype=jax.numpy.int16
        )

        other_vehicles_local_times = jax.numpy.zeros(
            (self._num_vehicles, self._num_vehicles - 1)
        )
        other_vehicles_capacities = jax.numpy.zeros(
            (self._num_vehicles, self._num_vehicles - 1), dtype=jax.numpy.int16
        )

        action_mask = jax.numpy.ones(
            (self._num_vehicles, self._num_customers + 1), dtype=jax.numpy.bool_
        )

        def mask_index(arr: chex.Array, ind: int) -> chex.Array:
            n = arr.shape[0]
            indices = jax.numpy.arange(n - 1) + (jax.numpy.arange(n - 1) >= ind)
            return arr[indices]

        mask_indices = jax.vmap(mask_index, in_axes=(None, 0))
        other_vehicles_positions = mask_indices(
            state.vehicles.positions, jax.numpy.arange(self._num_vehicles)
        )
        other_vehicles_capacities = mask_indices(
            state.vehicles.capacities, jax.numpy.arange(self._num_vehicles)
        )
        other_vehicles_local_times = mask_indices(
            state.vehicles.local_times, jax.numpy.arange(self._num_vehicles)
        )

        # The action is valid if the node has a
        # non-zero demand and the vehicle has enough capacity.
        def single_vehicle_action_mask(capacity: chex.Array) -> chex.Array:
            return (capacity >= state.nodes.demands) & (state.nodes.demands > 0.0)

        action_mask = jax.vmap(single_vehicle_action_mask, in_axes=(None, 0))(
            state.vehicles.capacities
        )

        # The depot is always a valid action (True).
        action_mask = action_mask.at[:, DEPOT_IDX].set(True)

        return Observation(
            nodes=Node(
                coordinates=jax.numpy.tile(
                    state.nodes.coordinates, (self._num_vehicles, 1, 1)
                ),
                demands=jax.numpy.tile(state.nodes.demands, (self._num_vehicles, 1)),
            ),
            windows=TimeWindow(
                start=jax.numpy.tile(state.windows.start, (self._num_vehicles, 1)),
                end=jax.numpy.tile(state.windows.end, (self._num_vehicles, 1)),
            ),
            coeffs=PenalityCoeff(
                early=jax.numpy.tile(state.coeffs.early, (self._num_vehicles, 1)),
                late=jax.numpy.tile(state.coeffs.late, (self._num_vehicles, 1)),
            ),
            other_vehicles=ObsVehicle(
                positions=other_vehicles_positions,
                local_times=other_vehicles_local_times,
                capacities=other_vehicles_capacities,
            ),
            main_vehicles=ObsVehicle(
                positions=state.vehicles.positions,
                local_times=state.vehicles.local_times,
                capacities=state.vehicles.capacities,
            ),
            action_mask=action_mask,
        )

    def _state_to_timestep(self, state: State) -> TimeStep:
        """
        Checks if the state is terminal and converts it into a timestep.

        Args:
            state: State object containing the dynamics of the environment.

        Returns:
            timestep: TimeStep object containing the timestep of the environment.
        """

        observation = self._state_to_observation(state)
        is_done = (state.nodes.demands.sum() == 0) & (
            state.vehicles.positions == DEPOT_IDX
        ).all() | jax.numpy.any(state.step_count > self._num_customers * 2)
        reward = self._reward_fn(state, is_done)

        timestep: TimeStep = jax.lax.cond(
            is_done,
            termination,
            transition,
            reward,
            observation,
        )
        return timestep
