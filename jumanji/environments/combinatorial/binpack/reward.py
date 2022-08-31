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

import jax
import jax.numpy as jnp
from typing_extensions import Protocol

from jumanji.environments.combinatorial.binpack.types import State, item_volume
from jumanji.types import Action


class RewardFn(Protocol):
    """Callable specification for the reward function."""

    def __call__(
        self,
        state: State,
        next_state: State,
        action: Action,
        done: jnp.bool_,
    ) -> jnp.float_:
        """The reward function used in the environment.

        Args:
            state: BinPack state before taking the action.
            next_state: BinPack state after taking the action.
            action: action taken by the agent to reach this state.
            done: whether the state is terminal.

        Returns:
            reward
        """


def sparse_linear_reward(
    state: State, next_state: State, action: Action, done: jnp.bool_
) -> jnp.float_:
    """Computes a sparse reward at the end of the episode. Returns volume utilization of the
    container (between 0.0 and 1.0).

    Args:
        state: BinPack state before taking the action.
        next_state: BinPack state after taking the action.
        action: action taken by the agent to reach this state.
        done: whether the state is terminal.

    Returns:
        linear sparse reward
    """
    del state, action

    def sparse_reward(state: State) -> jnp.float_:
        """Returns volume utilization between 0.0 and 1.0."""
        items_volume = jnp.sum(item_volume(state.items) * state.items_placed)
        container_volume = state.container.volume()
        return items_volume / container_volume

    reward = jax.lax.cond(
        done,
        sparse_reward,
        lambda _: jnp.array(0, float),
        next_state,
    )
    return reward
