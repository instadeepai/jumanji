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
import jax
import jax.numpy as jnp

from jumanji.environments.packing.flat_pack.types import State


class RewardFn(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        state: State,
        action: chex.Numeric,
        next_state: State,
        is_valid: bool,
        is_done: bool,
    ) -> chex.Numeric:
        """Compute the reward based on the current state, the chosen action,
        whether the action is valid and whether the episode is terminated.
        """


class DenseReward(RewardFn):
    """Reward function for the dense reward setting.

    This reward returns the number of non-zero cells in a placed block divided
        by the total number of cells in the grid. This means that the maximum possible
        episode return is 1.
    """

    def __call__(
        self,
        state: State,
        action: chex.Numeric,
        next_state: State,
        is_valid: bool,
        is_done: bool,
    ) -> chex.Numeric:
        """Compute the reward based on the current state, the chosen action,
        whether the action is valid and whether the episode is terminated.

        Note here, that the action taken is not the raw action received from the
        agent, but the block the agent opted to place on the grid.
        """

        del is_done
        del next_state
        del state

        action_ones = action != 0.0
        num_rows, num_cols = action_ones.shape

        reward = jax.lax.cond(
            is_valid,
            lambda: jnp.sum(action_ones, dtype=jnp.float32) / (num_rows * num_cols),
            lambda: jnp.float32(0.0),
        )

        return reward


class SparseReward(RewardFn):
    """Reward function for the sparse reward setting.

    This reward will return 0 at each timestep except for when all possible blocks have been
        placed on the grid in which case it will return 1.
    """

    def __call__(
        self,
        state: State,
        action: chex.Numeric,
        next_state: State,
        is_valid: bool,
        is_done: bool,
    ) -> chex.Numeric:
        """Compute the reward based on the current state, the chosen action,
        the next state, whether the action is valid and whether the episode is terminated.

        Note here, that the action taken is not the raw action received from the
        agent, but the block the agent opted to place on the grid.
        """

        del action
        del state

        completed_correctly = (
            is_done & jnp.all(next_state.current_grid != 0.0) & is_valid
        )

        reward = jax.lax.cond(
            completed_correctly,
            lambda: jnp.float32(1.0),
            lambda: jnp.float32(0.0),
        )

        return reward
