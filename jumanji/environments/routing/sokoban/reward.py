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

from jumanji.environments.routing.sokoban.constants import (
    BOX,
    LEVEL_COMPLETE_BONUS,
    N_BOXES,
    SINGLE_BOX_BONUS,
    STEP_BONUS,
    TARGET,
)
from jumanji.environments.routing.sokoban.types import State


class RewardFn(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        state: State,
        action: chex.Numeric,
        next_state: State,
    ) -> chex.Numeric:
        """Compute the reward based on the current state,
        the chosen action, the next state.
        """

    def count_targets(self, state: State) -> chex.Array:
        """
        Calculates the number of boxes on targets.

        Args:
            state: `State` object representing the current state of the
            environment.

        Returns:
            n_targets: Array (int32) of shape () specifying the number of boxes
            on targets.
        """

        mask_box = state.variable_grid == BOX
        mask_target = state.fixed_grid == TARGET

        num_boxes_on_targets = jnp.sum(mask_box & mask_target)

        return num_boxes_on_targets


class SparseReward(RewardFn):
    def __call__(
        self,
        state: State,
        action: chex.Array,
        next_state: State,
    ) -> chex.Array:
        """
        Implements the sparse reward function in the Sokoban environment.

        Args:
            state: `State` object The current state of the environment.
            action:  Array (int32) shape () representing the action taken.
            next_state:  `State` object The next state of the environment.

        Returns:
            reward: Array (float32) of shape () specifying the reward received
            at transition
        """

        next_num_box_target = self.count_targets(next_state)

        level_completed = next_num_box_target == N_BOXES

        return LEVEL_COMPLETE_BONUS * level_completed


class DenseReward(RewardFn):
    def __call__(
        self,
        state: State,
        action: chex.Array,
        next_state: State,
    ) -> chex.Array:
        """
        Implements the dense reward function in the Sokoban environment.

        Args:
            state: `State` object The current state of the environment.
            action:  Array (int32) shape () representing the action taken.
            next_state:  `State` object The next state of the environment.

        Returns:
            reward: Array (float32) of shape () specifying the reward received
            at transition
        """

        num_box_target = self.count_targets(state)
        next_num_box_target = self.count_targets(next_state)

        level_completed = next_num_box_target == N_BOXES

        return (
            SINGLE_BOX_BONUS * (next_num_box_target - num_box_target)
            + LEVEL_COMPLETE_BONUS * level_completed
            + STEP_BONUS
        )
