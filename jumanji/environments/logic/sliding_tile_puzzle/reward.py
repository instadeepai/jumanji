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

import abc

import chex
import jax.numpy as jnp

from jumanji.environments.logic.sliding_tile_puzzle.types import State


class RewardFn(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        state: State,
        action: chex.Numeric,
        next_state: State,
        solved_puzzle: chex.Array,
    ) -> chex.Numeric:
        """Compute the reward based on the current state, the chosen action, the next state,
        and the solved puzzle state."""


class SparseRewardFn(RewardFn):
    """Reward function that provides a sparse reward, only rewarding when the puzzle is solved."""

    def __call__(
        self,
        state: State,
        action: chex.Numeric,
        next_state: State,
        solved_puzzle: chex.Array,
    ) -> chex.Numeric:
        """
        Calculates the reward for the given state and action.

        Args:
            state: The current state.
            action: The chosen action.
            next_state: The state resulting from the chosen action.
            solved_puzzle: What the solved puzzle looks like.

        Returns:
            The calculated reward.
        """
        # The sparse reward is 1 if the puzzle is solved, and 0 otherwise.
        return jnp.array_equal(next_state.puzzle, solved_puzzle).astype(float)


class DenseRewardFn(RewardFn):
    """Reward function that provides a dense reward based on
    the difference of correctly placed tiles between states."""

    def __call__(
        self,
        state: State,
        action: chex.Numeric,
        next_state: State,
        solved_puzzle: chex.Array,
    ) -> chex.Numeric:
        """
        Calculates a dense reward for the given state and action. This dense
        reward is positive for each newly correctly placed tile and negative
        for each newly incorrectly placed tile.

        Args:
            state: The current state.
            action: The chosen action.
            next_state: The state resulting from the chosen action.
            solved_puzzle: What the solved puzzle looks like.

        Returns:
            The calculated dense reward.
        """
        new_correct_tiles = jnp.sum(
            (next_state.puzzle == solved_puzzle) & (state.puzzle != solved_puzzle)
        )
        new_incorrect_tiles = jnp.sum(
            (next_state.puzzle != solved_puzzle) & (state.puzzle == solved_puzzle)
        )
        return (new_correct_tiles - new_incorrect_tiles).astype(float)
