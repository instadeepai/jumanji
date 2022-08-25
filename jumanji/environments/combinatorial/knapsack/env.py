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
from chex import Array, PRNGKey
from jax import random

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.combinatorial.knapsack.specs import ObservationSpec
from jumanji.environments.combinatorial.knapsack.types import Observation, State
from jumanji.environments.combinatorial.knapsack.utils import (
    compute_value_items,
    generate_first_item,
    generate_problem,
)
from jumanji.types import Action, TimeStep, restart, termination, transition


class Knapsack(Environment[State]):
    """
    Knapsack environment as described in [1].
    - observation: Observation
        - problem: jax array (float32) of shape (problem_size, 2)
            the weights and values of the items.
        - first_item: jax array (int32)
            index of first added item (useless, but [1] does it to match TSP environment)
        - last_item: jax array (int32)
            index of the last added item (useless, but [1] does it to match TSP environment)
        - invalid_mask: jax array (int8) of shape (problem_size,)
            binary mask (0/1 <--> legal/illegal)

    - reward: jax array (float)
        the sum of the values of the items put in the bag at the end of the episode

    - episode termination:
        - if no action can be performed (all items remaining are larger than the bag capacity)
        - if an illegal action is performed (item is larger than the bag capacity or already taken)

    - state: State
    the state of the environment.
        - problem: : jax array (float32) of shape (problem_size, 2)
            the weights and values of the items.
        - last_item: jax array (int32)
            index of the last added item (useless, but [1] does it to match TSP environment)
        - first item: jax array (int32)
            index of the first added item (useless, but Pomo paper does it to match TSP setting)
        - used_mask:: jax array (int8) of shape (problem_size,)
            binary mask (0/1 <--> unused/used)
        - num_steps: jax array (int32)
            how many steps have been taken
        - remaining_budget: jax array (float32)
            the budget currently remaining

    [1] https://arxiv.org/abs/2010.16011
    """

    def __init__(self, problem_size: int = 10, total_budget: float = 2):
        self.problem_size = problem_size
        self.total_budget = total_budget

    def __repr__(self) -> str:
        return (
            f"0-1 Knapsack environment with {self.problem_size} items, "
            f"and a total budget of {self.total_budget}"
        )

    def reset(
        self,
        key: PRNGKey,
    ) -> Tuple[State, TimeStep]:
        """Resets the environment.

        Args:
            key: useful to pick the first item randomly.

        Returns:
            state: State object corresponding to the new state of the environment,
            timestep: TimeStep object corresponding the first timestep returned by the environment,
        """
        problem_key, start_key = random.split(key)
        problem = generate_problem(problem_key, self.problem_size)
        first_item = generate_first_item(start_key, self.problem_size)
        return self.reset_from_state(problem, first_item)

    def reset_from_state(
        self,
        problem: Array,
        first_item: jnp.int32,
    ) -> Tuple[State, TimeStep]:
        """Resets the environment from a given problem instance and first item.

        Args:
            problem: jax array (float32) of shape (problem_size, 2) the weights
                and values of the items.
            first_item: jax array (int32) index of the first added item
                (useless, but [1] does it to match TSP environment).

        Returns:
            state: State object corresponding to the new state of the environment.
            timestep: TimeStep object corresponding the first timestep returned by the environment.
        """
        state = State(
            problem=problem,
            last_item=None,
            first_item=first_item,
            used_mask=jnp.zeros(self.problem_size, dtype=jnp.int8),
            num_steps=jnp.int32(0),
            remaining_budget=jnp.float32(self.total_budget),
        )
        state = self._update_state(state, first_item)
        timestep = restart(observation=self._state_to_observation(state))
        return state, timestep

    def step(self, state: State, action: Action) -> Tuple[State, TimeStep]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the index of next item to take.

        Returns:
            state: next state of the environment.
            timestep: the timestep to be observed.
        """

        state_budget_fits = state.remaining_budget >= state.problem[action, 0]
        new_item = state.used_mask[action] == 0
        is_valid = state_budget_fits & new_item
        state = jax.lax.cond(
            pred=is_valid,
            true_fun=lambda x: self._update_state(x[0], x[1]),
            false_fun=lambda _: state,
            operand=[state, action],
        )
        timestep = self._state_to_timestep(state, is_valid)
        return state, timestep

    def observation_spec(self) -> ObservationSpec:
        """Returns the observation spec.

        Returns:
            observation_spec: a tree of specs containing the spec for each of the constituent fields
                of an observation.
        """
        problem_obs = specs.BoundedArray(
            shape=(self.problem_size, 2),
            minimum=0.0,
            maximum=1.0,
            dtype=jnp.float32,
            name="problem",
        )
        first_item_obs = specs.DiscreteArray(
            self.problem_size, dtype=jnp.int32, name="first_item"
        )
        last_item_obs = specs.DiscreteArray(
            self.problem_size, dtype=jnp.int32, name="last_item"
        )
        invalid_mask = specs.BoundedArray(
            shape=(self.problem_size,),
            dtype=jnp.int8,
            minimum=0,
            maximum=1,
            name="invalid_mask",
        )
        return ObservationSpec(
            problem_obs,
            first_item_obs,
            last_item_obs,
            invalid_mask,
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec.

        Returns:
            action_spec: a `specs.DiscreteArray` spec.
        """
        return specs.DiscreteArray(self.problem_size, name="action")

    def _update_state(self, state: State, next_item: int) -> State:
        """Updates the state of the environment.

        Args:
            state: State object containing the dynamics of the environment.
            next_item: int, index of the next item to take.

        Returns:
            state: State object corresponding to the new state of the environment.
        """
        return State(
            problem=state.problem,
            last_item=next_item,
            first_item=state.first_item,
            used_mask=state.used_mask.at[next_item].set(1),
            num_steps=state.num_steps + 1,
            remaining_budget=state.remaining_budget - state.problem[next_item, 0],
        )

    def _state_to_observation(self, state: State) -> Observation:
        """Converts a state to an observation.

        Args:
            state: State object containing the dynamics of the environment.

        Returns:
            observation: Observation object containing the observation of the environment.
        """

        return Observation(
            problem=state.problem,
            first_item=state.first_item,
            last_item=state.last_item,
            invalid_mask=state.used_mask
            | (state.remaining_budget < state.problem[:, 0]),
        )

    def _state_to_timestep(self, state: State, is_valid: bool) -> TimeStep:
        """Checks if the state is terminal and converts it to a timestep.

        Args:
            state: State object containing the dynamics of the environment.
            is_valid: Boolean indicating whether the last action was valid.

        Returns:
            timestep: TimeStep object containing the timestep of the environment.
                The episode terminates if there is no legal item to take or if
                the last action was invalid.
        """
        is_done = (
            jnp.min(jnp.where(state.used_mask == 0, state.problem[:, 0], 1))
            > state.remaining_budget
        ) | (~is_valid)

        def make_termination_timestep(state: State) -> TimeStep:
            return termination(
                reward=compute_value_items(state.problem, state.used_mask),
                observation=self._state_to_observation(state),
            )

        def make_transition_timestep(state: State) -> TimeStep:
            return transition(
                reward=jnp.float32(0), observation=self._state_to_observation(state)
            )

        timestep: TimeStep = jax.lax.cond(
            is_done,
            make_termination_timestep,
            make_transition_timestep,
            state,
        )

        return timestep
