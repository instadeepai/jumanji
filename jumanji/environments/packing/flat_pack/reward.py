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
        placed_block: chex.Numeric,
        next_state: State,
        is_valid: bool,
        is_done: bool,
    ) -> chex.Numeric:
        """Compute the reward based on the current state, the chosen action,
        whether the action is valid and whether the episode is terminated.
        """


class CellDenseReward(RewardFn):
    """Reward function for the dense reward setting.

    This reward returns the number of non-zero cells in a placed block normalised
        by the total number of cells in the grid. This means that the maximum possible
        episode return is 1. That is to say that, in the case of this reward, an agent
        will optimise for maximal area coverage in the the grid.
    """

    def __call__(
        self,
        state: State,
        placed_block: chex.Numeric,
        next_state: State,
        is_valid: bool,
        is_done: bool,
    ) -> chex.Numeric:
        """Compute the reward based on the current state, the chosen action,
        whether the action is valid and whether the episode is terminated.
        """

        del is_done
        del next_state
        del state

        num_rows, num_cols = placed_block.shape

        reward = jax.lax.cond(
            is_valid,
            lambda: jnp.sum(placed_block != 0.0, dtype=jnp.float32)
            / (num_rows * num_cols),
            lambda: jnp.float32(0.0),
        )

        return reward


class BlockDenseReward(RewardFn):
    """Reward function for the dense reward setting.

    This reward will give a normalised reward for each block placed on the grid
    with each block being equally weighted. This implies that each placed block
    will have a reward of `1 / num_blocks` and the maximum possible episode return
    is 1. That is to say that, in the case of this reward, an agent will optimise
    for placing as many blocks as possible on the grid.
    """

    def __call__(
        self,
        state: State,
        placed_block: chex.Numeric,
        next_state: State,
        is_valid: bool,
        is_done: bool,
    ) -> chex.Numeric:
        """Compute the reward based on the current state, the chosen action,
        whether the action is valid and whether the episode is terminated.
        """

        del is_done
        del next_state
        del placed_block

        num_blocks = state.num_blocks
        del state

        reward = jax.lax.cond(
            is_valid,
            lambda: jnp.float32(1.0 / num_blocks),
            lambda: jnp.float32(0.0),
        )

        return reward
