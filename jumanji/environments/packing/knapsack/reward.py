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

from jumanji.environments.packing.knapsack.types import State


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
        """Compute the reward based on the current state, the chosen action, the next state,
        whether the action is valid and whether the episode is terminated.
        """


class SparseReward(RewardFn):
    """Sparse reward computed by summing the values of the items packed in the bag at the end
    of the episode. The reward is 0 if the episode is not terminated yet or if the action is
    invalid, i.e. an item that was previously selected is selected again or has a weight larger
    than the bag capacity.
    """

    def __call__(
        self,
        state: State,
        action: chex.Array,
        next_state: State,
        is_valid: bool,
        is_done: bool,
    ) -> chex.Numeric:
        reward = jax.lax.cond(
            is_done & is_valid,
            jnp.dot,
            lambda *_: jnp.array(0, float),
            next_state.packed_items,
            next_state.values,
        )
        return reward


class DenseReward(RewardFn):
    """Dense reward corresponding to the value of the item to pack at the current timestep.
    The reward is 0 if the episode is not terminated yet or if the action is invalid, i.e. an item
    that was previously selected is selected again or has a weight larger than the bag capacity.
    """

    def __call__(
        self,
        state: State,
        action: chex.Array,
        next_state: State,
        is_valid: bool,
        is_done: bool,
    ) -> chex.Numeric:
        del is_done
        reward = jax.lax.select(
            is_valid,
            state.values[action],
            jnp.array(0, float),
        )
        return reward
