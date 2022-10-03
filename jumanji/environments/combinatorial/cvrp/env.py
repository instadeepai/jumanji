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

from typing import Tuple

import jax
import jax.numpy as jnp
from chex import PRNGKey
from jax import random

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.combinatorial.tsp.utils import compute_tour_length
from jumanji.environments.combinatorial.cvrp.specs import ObservationSpec
from jumanji.environments.combinatorial.cvrp.types import Observation, State
from jumanji.environments.combinatorial.cvrp.utils import (
    DEPOT_IDX,
    generate_problem
)

from jumanji.types import Action, TimeStep, termination, transition, restart


class CVRP(Environment[State]):
    """
    Capacitated Vehicle Routing Problem (CVRP) environment as described in [1].
    - observation: Observation
        - problem: jax array (float32) of shape (num_nodes + 1, 3)
            the coordinates of each node and the depot, and the associated cost (0.0 for the depot)
        - position: jax array (float32)
            the index of the last visited node
        - capacity: jax array (float32)
            the current capacity of the vehicle
        - action_mask: jax array (int8) of shape (num_nodes + 1,)
            binary mask (0/1 <--> visitable/not visitable)

    - reward: jax array (float32)
        the sum of the distances between consecutive nodes at the end of the episode (the reward is 0 if a previously
        selected non-dept node is selected again, or the depot is selected twice in a row)

    - state: State
        - problem: jax array (float32) of shape (num_nodes + 1, 3)
            the coordinates of each node and the depot, and the associated cost (0.0 for the depot)
        - position: jax array (float32)
            the index of the last visited node
        - capacity: jax array (float32)
            the current capacity of the vehicle
        - visited_mask: jax array (int8) of shape (num_nodes,)
            binary mask (0/1 <--> not visited/visited)
        - order: jax array (int32) of shape (2 * num_nodes,)
            the identifiers of the nodes that have been visited (-1 means that no node has been visited yet at that
            time in the sequence)
        - num_visits: int32
            number of actions that have been taken (i.e., unique visits)

    [1] Toth P., Vigo D. (2014). "Vehicle routing: problems, methods, and applications".
    """

    def __init__(self, problem_size: int = 100, max_capacity: int = 30, max_demand: int = 10):
        assert max_capacity >= max_demand, (
            f"The demand associated with each node must be lower than the maximum capacity, hence the "
            f"maximum capacity must be >= {max_demand}."
        )
        self.problem_size = problem_size
        self.max_capacity = max_capacity
        self.max_demand = max_demand

    def __repr__(self):
        return f"CVRP(problem_size={self.problem_size}, max_capacity={self.max_capacity}, max_demand={self.max_demand})"

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep]:
        """
        Resets the environment.

        Args:
            key: used to randomly generate the problem and the start node.

        Returns:
             state: State object corresponding to the new state of the environment.
             timestep: TimeStep object corresponding to the first timestep returned by the environment.
        """
        problem_key, start_key = random.split(key)
        problem = generate_problem(problem_key, self.problem_size, self.max_demand)
        state = State(
            problem=problem,
            position=jnp.int32(DEPOT_IDX),
            capacity=jnp.float32(self.max_capacity),
            visited_mask=jnp.zeros(self.problem_size + 1, dtype=jnp.int8).at[DEPOT_IDX].set(1),
            order=jnp.zeros(self.get_episode_horizon(), jnp.int32),
            num_total_visits=jnp.int32(1),
        )
        timestep = restart(observation=self._state_to_observation(state))
        return state, timestep

    def step(self, state: State, action: Action) -> Tuple[State, TimeStep]:
        """
        Run one timestep of the environment's dynamics.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the index of the next node to visit.

        Returns:
            state, timestep: Tuple[State, TimeStep] containing the next state of the environment, as well
            as the timestep to be observed.
        """
        is_valid = (state.visited_mask[action] == 0) & (state.capacity >= state.problem[action, -1])

        state = jax.lax.cond(
            pred=is_valid,
            true_fun=lambda x: self._update_state(x[0], x[1]),
            false_fun=lambda _: state,
            operand=[state, action],
        )
        timestep = self._state_to_timestep(state, is_valid)
        return state, timestep

    def observation_spec(self) -> ObservationSpec:
        """
        Returns the observation spec.

        Returns:
            observation_spec: a Tuple containing the spec for each of the constituent fields of an observation.
        """
        problem_obs = specs.BoundedArray(
            shape=(self.problem_size + 1, 3),
            minimum=0.0,
            maximum=1.0,
            dtype=jnp.float32,
            name="problem",
        )
        position_obs = specs.DiscreteArray(
            self.problem_size + 1, dtype=jnp.int32, name="position"
        )
        capacity_obs = specs.BoundedArray(
            shape=(), minimum=0.0, maximum=30.0, dtype=jnp.float32, name="capacity"
        )
        action_mask = specs.BoundedArray(
            shape=(self.problem_size + 1,),
            dtype=jnp.int8,
            minimum=0,
            maximum=1,
            name="action mask",
        )
        return ObservationSpec(problem_obs, position_obs, capacity_obs, action_mask)

    def action_spec(self) -> specs.Array:
        """
        Returns the action spec.

        Returns:
            action_spec: a `dm_env.specs.Array` spec.
        """
        return specs.DiscreteArray(self.problem_size + 1, name="action")

    def _update_state(self, state: State, next_node: jnp.int32) -> State:
        """
        Updates the state of the environment.

        Args:
            state: State object containing the dynamics of the environment.
            next_node: int, index of the next node to visit.

        Returns:
            state: State object corresponding to the new state of the environment.
        """
        is_not_depot = jnp.int32(next_node != DEPOT_IDX)

        next_node = jax.lax.cond(
            pred=state.visited_mask.sum() == self.problem_size + 1,
            true_fun=lambda _: DEPOT_IDX,  # stay in the depot if we have visited all nodes
            false_fun=lambda _: next_node,
            operand=None,
        )

        # Set depot to 0 (valid to visit) since it can be visited multiple times
        visited_mask = state.visited_mask.at[DEPOT_IDX].set(0)

        return State(
            problem=state.problem,
            position=next_node,
            capacity=is_not_depot * (state.capacity - jnp.int32(state.problem[next_node, 2])) + (
                        1 - is_not_depot) * self.max_capacity,
            visited_mask=visited_mask.at[next_node].set(1),
            order=state.order.at[state.num_total_visits].set(next_node),
            num_total_visits=state.num_total_visits + 1,
        )

    def _state_to_observation(self, state: State) -> Observation:
        """
        Converts a state into an observation.

        Args:
            state: State object containing the dynamics of the environment.

        Returns:
            observation: Observation object containing the observation of the environment.
        """
        # A node is invalid if it has been visited or the vehicle does not have enough capacity to cover its demand.
        action_mask = state.visited_mask | (state.capacity < state.problem[:, 2])
        # The depot is valid (0) if we are not at it, else it is invalid (1).
        action_mask = action_mask.at[DEPOT_IDX].set(jnp.int32(state.position == DEPOT_IDX))

        return Observation(
            problem=state.problem.at[:, -1].set(jnp.float32(state.problem[:, -1] / self.max_capacity)),
            position=state.position,
            capacity=state.capacity,
            action_mask=action_mask,
        )

    def _state_to_timestep(self, state: State, is_valid: bool) -> TimeStep:
        """
        Checks if the state is terminal and converts it into a timestep.

        Args:
            state: State object containing the dynamics of the environment.

        Returns:
            timestep: TimeStep object containing the timestep of the environment.
        """

        def make_termination_timestep(state: State) -> TimeStep:
            return termination(
                reward=-compute_tour_length(state.problem[:, :2], state.order),
                observation=self._state_to_observation(state),
            )

        def make_transition_timestep(state: State) -> TimeStep:
            return transition(
                reward=jnp.float32(0), observation=self._state_to_observation(state)
            )

        is_done = (state.visited_mask.sum() == self.problem_size + 1) | (~is_valid)
        return jax.lax.cond(
            is_done,
            make_termination_timestep,
            make_transition_timestep,
            state,
        )

    def get_episode_horizon(self):
        return 2 * self.problem_size
