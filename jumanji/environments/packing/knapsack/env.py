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

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.packing.knapsack.types import Observation, State
from jumanji.environments.packing.knapsack.utils import (
    compute_value_items,
    generate_problem,
)
from jumanji.types import Action, TimeStep, restart, termination, transition


class Knapsack(Environment[State]):
    """
    Knapsack environment as described in [1].
    - observation: Observation
        - weights: jax array (float32) of shape (num_items,)
            the weights of the items.
        - values: jax array (float32) of shape (num_items,)
            the values of the items.
        - action_mask: jax array (bool) of shape (num_items,)
            binary mask (False/True <--> invalid/valid).

    - reward: jax array (float)
        the sum of the values of the items put in the bag at the end of the episode.

    - episode termination:
        - if no action can be performed (all items remaining are larger than the bag capacity).
        - if an illegal action is performed (item is larger than the bag capacity or already taken).

    - state: State
    the state of the environment.
        - weights: jax array (float32) of shape (num_items,)
            the weights of the items.
        - values: jax array (float32) of shape (num_items,)
            the values of the items.
        - packed_items: jax array (bool) of shape (num_items,)
            binary mask indicating if an item is in the knapsack or not (False/True <--> out/in).
        - remaining_budget: jax array (float32)
            the budget currently remaining.

    [1] https://arxiv.org/abs/2010.16011
    """

    def __init__(self, num_items: int = 10, total_budget: float = 2):
        self.num_items = num_items
        self.total_budget = total_budget

    def __repr__(self) -> str:
        return (
            f"0-1 Knapsack environment with {self.num_items} items, "
            f"and a total budget of {self.total_budget}"
        )

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep]:
        """Resets the environment.

        Args:
            key: used to randomly generate the weights abd values of the items.

        Returns:
            state: the new state of the environment.
            timestep: the first timestep returned by the environment.
        """
        weights, values = generate_problem(key, self.num_items)
        state = State(
            weights=weights,
            values=values,
            packed_items=jnp.zeros(self.num_items, dtype=bool),
            remaining_budget=jnp.float32(self.total_budget),
        )
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
        state_budget_fits = state.remaining_budget >= state.weights[action]
        new_item = state.packed_items[action] == 0
        is_valid = state_budget_fits & new_item
        state = jax.lax.cond(
            pred=is_valid,
            true_fun=lambda x: self._update_state(x[0], x[1]),
            false_fun=lambda _: state,
            operand=[state, action],
        )
        timestep = self._state_to_timestep(state, is_valid)
        return state, timestep

    def observation_spec(self) -> specs.Spec:
        """Specifications of the observation of the `Knapsack` environment.

        Returns:
            Spec containing all the specifications for all the `Observation` fields.
        """
        weights = specs.BoundedArray(
            shape=(self.num_items,),
            minimum=0.0,
            maximum=1.0,
            dtype=jnp.float32,
            name="weights",
        )
        values = specs.BoundedArray(
            shape=(self.num_items,),
            minimum=0.0,
            maximum=1.0,
            dtype=jnp.float32,
            name="values",
        )
        action_mask = specs.BoundedArray(
            shape=(self.num_items,),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            weights=weights,
            values=values,
            action_mask=action_mask,
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec.

        Returns:
            action_spec: a `specs.DiscreteArray` spec.
        """
        return specs.DiscreteArray(self.num_items, name="action")

    def _update_state(self, state: State, next_item: int) -> State:
        """Updates the state of the environment.

        Args:
            state: State object containing the dynamics of the environment.
            next_item: int, index of the next item to take.

        Returns:
            state: State object corresponding to the new state of the environment.
        """
        return State(
            weights=state.weights,
            values=state.values,
            packed_items=state.packed_items.at[next_item].set(True),
            remaining_budget=state.remaining_budget - state.weights[next_item],
        )

    def _state_to_observation(self, state: State) -> Observation:
        """Converts a state to an observation.

        Args:
            state: State object containing the dynamics of the environment.

        Returns:
            observation: Observation object containing the observation of the environment.
        """

        return Observation(
            weights=state.weights,
            values=state.values,
            action_mask=jnp.logical_not(
                state.packed_items | (state.remaining_budget < state.weights)
            ),
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
            jnp.min(jnp.where(state.packed_items == 0, state.weights, 1))
            > state.remaining_budget
        ) | (~is_valid)

        def make_termination_timestep(state: State) -> TimeStep:
            return termination(
                reward=compute_value_items(state.values, state.packed_items),
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
