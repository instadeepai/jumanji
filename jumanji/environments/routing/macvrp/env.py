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

from typing import Optional, Tuple

import jax
import chex
import numpy as np
from numpy.typing import NDArray

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.macvrp.env_viewer import MACVRPViewer
from jumanji.viewer import Viewer
from jumanji.environments.routing.macvrp.reward import DenseReward, RewardFn
from jumanji.environments.routing.macvrp.specs import (
    NodeSpec,
    ObservationSpec,
    PenalityCoeffSpec,
    TimeWindowSpec,
    VehicleSpec,
)
from jumanji.environments.routing.macvrp.types import (
    Node,
    Observation,
    PenalityCoeff,
    State,
    TimeWindow,
    Vehicle,
)
from jumanji.environments.routing.macvrp.utils import (
    DEPOT_IDX,
    compute_distance,
    compute_time_penalties,
    generate_problem,
    get_init_settings,
)
from jumanji.types import Action, TimeStep, restart, termination, transition


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


    # Use this
    """Instantiates a `CVRP` environment.

        Args:
            num_nodes: number of city nodes in the environment. Defaults to 20.
            max_capacity: maximum capacity of the vehicle. Defaults to 30.
            max_demand: maximum demand of each node. Defaults to 10.
            reward_fn: `RewardFn` whose `__call__` method computes the reward of an environment
                transition. The function must compute the reward based on the current state,
                the chosen action, the next state and whether the action is valid.
                Implemented options are [`DenseReward`, `SparseReward`]. Defaults to `DenseReward`.
            viewer: `Viewer` used for rendering. Defaults to `CVRPViewer` with "human" render mode.
        """

    def __init__(
        self,
        num_customers: int = 20,
        num_vehicles: int = 2,
        render_mode: str = "human",
        reward_fn: Optional[RewardFn] = None,
        viewer: Optional[Viewer] = None,
    ):
        """
        Instantiates a `MACVRP` environment.

        Args:
            num_customers: number of customer nodes in the environment. Defaults to 20.
            num_vehicles: number of vehicles in the environment. Defaults to 2.
            render_mode: render mode used by the viewer. Defaults to "human".
            reward_fn: `RewardFn` whose `__call__` method computes the reward of an environment
                transition. The function must compute the reward based on the current state 
                and whether the environment is done.
                Implemented options are [`DenseReward`]. Defaults to `DenseReward`.
            viewer: `Viewer` used for rendering. Defaults to `MACVRPViewer` with "human" render mode.
        """


        
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles

        # Scenario are taken from the paper

        # From the paper: All distances are represented by Euclidean distances in the plane,
        # and the speeds of all  vehicles are assumed to be identical (i.e., it takes one
        # unit of time to travel one unit of distance)
        self.speed = 1

        # Note: The time window detail could not be found in the paper for
        # the 20, 50 and 150 customer scenarios. We use the 150 customer scenario's
        # time window of 20 for them.
        self.time_window_length = 20

        (
            self.map_max,
            self.max_capacity,
            self.max_start_window,
            self.early_coef_rand,
            self.late_coef_rand,
            self.customer_demand_max,
        ) = get_init_settings(num_customers, num_vehicles)

        self.max_end_window = self.max_start_window + self.time_window_length

        self.max_local_time = 2.0 * jax.numpy.sqrt(2.0) * self.map_max * self.num_customers

        self.reward_fn = reward_fn or DenseReward()

        # Create viewer used for rendering
        self._env_viewer = viewer or MACVRPViewer(
            name="MACVRP",
            num_vehicles=self.num_vehicles,
            num_customers=self.num_customers,
            map_max=self.map_max,
            render_mode=render_mode,
        )

    def __repr__(self) -> str:
        return f"MACVRP(num_customers={self.num_customers}, num_vehicles={self.num_vehicles})"

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """
        Resets the environment.

        Args:
            key: used to randomly generate the problem and the start node.

        Returns:
             state: State object corresponding to the new state of the environment.
             timestep: TimeStep object corresponding to the first timestep returned by the
                environment.
        """
        # This split is uncessary, but it makes the code more readable.
        problem_key, _ = jax.random.split(key)

        total_capacity = self.max_capacity * self.num_vehicles
        (
            node_coordinates,
            node_demands,
            window_start_times,
            window_end_times,
            early_coefs,
            late_coefs,
        ) = generate_problem(
            problem_key,
            self.num_customers,
            total_capacity,
            self.map_max,
            self.customer_demand_max,
            self.max_start_window,
            self.time_window_length,
            self.early_coef_rand,
            self.late_coef_rand,
        )

        state = State(
            nodes=Node(coordinates=node_coordinates, demands=node_demands),
            windows=TimeWindow(start=window_start_times, end=window_end_times),
            coeffs=PenalityCoeff(early=early_coefs, late=late_coefs),
            vehicles=Vehicle(
                positions=np.int16([DEPOT_IDX] * self.num_vehicles),
                local_times=jax.numpy.zeros(self.num_vehicles, dtype=jax.numpy.float32),
                capacities=jax.numpy.ones(self.num_vehicles, dtype=jax.numpy.int16)
                * self.max_capacity,
                distances=jax.numpy.zeros(self.num_vehicles, dtype=jax.numpy.float32),
                time_penalties=jax.numpy.zeros(self.num_vehicles, dtype=jax.numpy.float32),
            ),
            order=jax.numpy.zeros(
                (self.num_vehicles, 2 * self.num_customers), dtype=jax.numpy.int16
            ),
            step_count=jax.numpy.ones((), dtype=np.int16),
        )

        timestep = restart(observation=self._state_to_observation(state))

        return state, timestep

    def step(self, state: State, actions: Action) -> Tuple[State, TimeStep]:
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

    def observation_spec(self) -> ObservationSpec:
        """
        Returns the observation spec.

        Returns:
            observation_spec: a Tuple containing the spec for each of the constituent fields of an
                observation.
        """

        node_coordinates = specs.BoundedArray(
            shape=(self.num_vehicles, self.num_customers + 1, 2),
            minimum=0.0,
            maximum=self.map_max,
            dtype=jax.numpy.float32,
            name="node_coordinates",
        )

        node_demands = specs.BoundedArray(
            shape=(
                self.num_vehicles,
                self.num_customers + 1,
            ),
            minimum=0,
            maximum=self.max_capacity,
            dtype=jax.numpy.int16,
            name="node_demands",
        )

        node_time_windows_start = specs.BoundedArray(
            shape=(self.num_vehicles, self.num_customers + 1),
            minimum=0.0,
            maximum=self.max_end_window,
            dtype=jax.numpy.float32,
            name="node_time_windows_start",
        )

        node_time_windows_end = specs.BoundedArray(
            shape=(self.num_vehicles, self.num_customers + 1),
            minimum=0.0,
            maximum=self.max_end_window,
            dtype=jax.numpy.float32,
            name="node_time_windows_end",
        )

        node_penalty_coeffs_start = specs.BoundedArray(
            shape=(self.num_vehicles, self.num_customers + 1),
            minimum=0.0,
            maximum=self.late_coef_rand[-1],
            dtype=jax.numpy.float32,
            name="node_penalty_coeffs_start",
        )

        node_penalty_coeffs_end = specs.BoundedArray(
            shape=(self.num_vehicles, self.num_customers + 1),
            minimum=0.0,
            maximum=self.late_coef_rand[-1],
            dtype=jax.numpy.float32,
            name="node_penalty_coeffs_end",
        )

        other_vehicles_positions = specs.BoundedArray(
            shape=(self.num_vehicles, self.num_vehicles - 1),
            minimum=0,
            maximum=self.num_customers + 1,
            dtype=jax.numpy.int16,
            name="other_vehicles_positions",
        )

        other_vehicles_local_times = specs.BoundedArray(
            shape=(
                self.num_vehicles,
                self.num_vehicles - 1,
            ),
            minimum=0.0,
            maximum=self.max_local_time,
            dtype=jax.numpy.float32,
            name="other_vehicles_local_times",
        )

        other_vehicles_capacities = specs.BoundedArray(
            shape=(
                self.num_vehicles,
                self.num_vehicles - 1,
            ),
            minimum=0,
            maximum=self.max_capacity,
            dtype=jax.numpy.int16,
            name="other_vehicles_capacities",
        )

        vehicle_position = specs.BoundedArray(
            shape=(self.num_vehicles,),
            minimum=0,
            maximum=self.num_customers + 1,
            dtype=jax.numpy.int16,
            name="vehicle_position",
        )

        vehicle_local_time = specs.BoundedArray(
            shape=(self.num_vehicles,),
            minimum=0.0,
            maximum=self.max_local_time,
            dtype=jax.numpy.float32,
            name="vehicle_local_time",
        )

        vehicle_capacity = specs.BoundedArray(
            shape=(self.num_vehicles,),
            minimum=0,
            maximum=self.max_capacity,
            dtype=jax.numpy.int16,
            name="vehicle_capacity",
        )

        action_mask = specs.BoundedArray(
            shape=(self.num_vehicles, self.num_customers + 1),
            minimum=0,
            maximum=self.num_customers + 1,
            dtype=jax.numpy.bool_,
            name="action_mask",
        )

        return ObservationSpec(
            nodes_spec=NodeSpec(
                coordinates_spec=node_coordinates,
                demands_spec=node_demands,
            ),
            windows_spec=TimeWindowSpec(
                start_spec=node_time_windows_start, end_spec=node_time_windows_end
            ),
            coeffs_spec=PenalityCoeffSpec(
                early_spec=node_penalty_coeffs_start, late_spec=node_penalty_coeffs_end
            ),
            other_vehicles_spec=VehicleSpec(
                positions_spec=other_vehicles_positions,
                local_times_spec=other_vehicles_local_times,
                capacities_spec=other_vehicles_capacities,
            ),
            main_vehicles_spec=VehicleSpec(
                positions_spec=vehicle_position,
                local_times_spec=vehicle_local_time,
                capacities_spec=vehicle_capacity,
            ),
            action_mask_spec=action_mask,
        )

    def action_spec(self) -> specs.BoundedArray:
        """
        Returns the action spec.

        Returns:
            action_spec: a `specs.BoundedArray` spec.
        """

        return specs.BoundedArray(
            (self.num_vehicles,), jax.numpy.int16, 0, self.num_customers + 1, "actions"
        )

    def render(self, state: State) -> Optional[NDArray]:
        """Render the given state of the environment. This rendering shows the layout of the tour so
            far with the cities as circles, and the depot as a square.

        Args:
            state: environment state to render.

        Returns:
            rgb_array: the RGB image of the state as an array.
        """
        return self._env_viewer.render(state)

    def _update_state(self, state: State, actions: Action) -> State:
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
            next_nodes, return_index=True, size=self.num_vehicles
        )
        new_nodes = jax.numpy.zeros(len(next_nodes), dtype=next_nodes.dtype)
        new_nodes = new_nodes.at[unique_indices].set(values)

        # Update the vehicle distances and local times.
        start_coords = state.nodes.coordinates[state.vehicles.positions]
        end_coords = state.nodes.coordinates[next_nodes]
        step_travel_distances = compute_distance(start_coords, end_coords)
        vehicle_distances = state.vehicles.distances + step_travel_distances
        vehicle_local_times = (
            state.vehicles.local_times + step_travel_distances / self.speed
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
            self.max_capacity,
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
            vehicles=Vehicle(
                local_times=vehicle_local_times,
                positions=next_nodes,
                capacities=vehicle_capacities,
                distances=vehicle_distances,
                time_penalties=vehicle_time_penalties,
            ),
            step_count=state.step_count + 1,
            order=order,
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
            (self.num_vehicles, self.num_vehicles - 1), dtype=jax.numpy.int16
        )

        other_vehicles_local_times = jax.numpy.zeros(
            (self.num_vehicles, self.num_vehicles - 1)
        )
        other_vehicles_capacities = jax.numpy.zeros(
            (self.num_vehicles, self.num_vehicles - 1), dtype=jax.numpy.int16
        )

        action_mask = jax.numpy.ones(
            (self.num_vehicles, self.num_customers + 1), dtype=jax.numpy.bool_
        )

        # Generate the other vehicle coordinates
        for i in range(self.num_vehicles):
            other_vehicles_positions = other_vehicles_positions.at[i].set(
                jax.numpy.concatenate(
                    [
                        state.vehicles.positions[:i],
                        state.vehicles.positions[i + 1 :],  # noqa
                    ]
                )
            )

            other_vehicles_local_times = other_vehicles_local_times.at[i].set(
                jax.numpy.concatenate(
                    [
                        state.vehicles.local_times[:i],
                        state.vehicles.local_times[i + 1 :],  # noqa
                    ]
                )
            )
            other_vehicles_capacities = other_vehicles_capacities.at[i].set(
                jax.numpy.concatenate(
                    [
                        state.vehicles.capacities[:i],
                        state.vehicles.capacities[i + 1 :],  # noqa
                    ]
                )
            )

            # The action is valid if the node has a
            # non-zero demand and the vehicle has enough capacity.
            action_mask = action_mask.at[i].set(
                (state.vehicles.capacities[i] >= state.nodes.demands)
                & (state.nodes.demands > 0.0)
            )

        # The depot is always a valid action (True).
        action_mask = action_mask.at[:, DEPOT_IDX].set(True)

        return Observation(
            nodes=Node(
                coordinates=jax.numpy.tile(
                    state.nodes.coordinates, (self.num_vehicles, 1, 1)
                ),
                demands=jax.numpy.tile(state.nodes.demands, (self.num_vehicles, 1)),
            ),
            windows=TimeWindow(
                start=jax.numpy.tile(state.windows.start, (self.num_vehicles, 1)),
                end=jax.numpy.tile(state.windows.end, (self.num_vehicles, 1)),
            ),
            coeffs=PenalityCoeff(
                early=jax.numpy.tile(state.coeffs.early, (self.num_vehicles, 1)),
                late=jax.numpy.tile(state.coeffs.late, (self.num_vehicles, 1)),
            ),
            other_vehicles=Vehicle(
                positions=other_vehicles_positions,
                local_times=other_vehicles_local_times,
                capacities=other_vehicles_capacities,
            ),
            main_vehicles=Vehicle(
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
        ).all() | np.any(state.step_count >= self.num_customers * 2)
        reward = self.reward_fn(state, is_done)

        timestep: TimeStep = jax.lax.cond(
            is_done,
            termination,
            transition,
            reward,
            observation,
        )
        return timestep
