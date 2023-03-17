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

from jumanji.environments.packing.bin_pack.types import State, item_volume
from jumanji.tree_utils import tree_slice


class RewardFn(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        state: State,
        action: chex.Array,
        next_state: State,
        is_valid: bool,
        is_done: bool,
    ) -> chex.Numeric:
        """Compute the reward based on the current state, the chosen action, the next state,
        whether the transition is valid and if it is terminal.
        """


class DenseReward(RewardFn):
    """Computes a reward at each timestep, equal to the normalized volume (relative to the container
    volume) of the item packed by taking the chosen action. The computed reward is equivalent
    to the increase in volume utilization of the container due to packing the chosen item.
    If the action is invalid, the reward is 0.0 instead.
    """

    def __call__(
        self,
        state: State,
        action: chex.Array,
        next_state: State,
        is_valid: bool,
        is_done: bool,
    ) -> float:
        del next_state, is_done
        _, item_id = action
        chosen_item_volume = item_volume(tree_slice(state.items, item_id))
        container_volume = state.container.volume()
        reward = chosen_item_volume / container_volume
        reward: float = jax.lax.select(is_valid, reward, jnp.array(0, float))
        return reward


class SparseReward(RewardFn):
    """Computes a sparse reward at the end of the episode. Returns the volume utilization of the
    container (between 0.0 and 1.0).
    If the action is invalid, the action is ignored and the reward is still returned as the current
    container utilization.
    """

    def __call__(
        self,
        state: State,
        action: chex.Array,
        next_state: State,
        is_valid: bool,
        is_done: bool,
    ) -> float:
        del state, action, is_valid

        def sparse_reward(state: State) -> jnp.float_:
            """Returns volume utilization between 0.0 and 1.0."""
            items_volume = jnp.sum(item_volume(state.items) * state.items_placed)
            container_volume = state.container.volume()
            return items_volume / container_volume

        reward: float = jax.lax.cond(
            is_done,
            sparse_reward,
            lambda _: jnp.array(0, float),
            next_state,
        )
        return reward
