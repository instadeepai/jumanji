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

from jumanji.environments.logic.minesweeper.types import State
from jumanji.environments.logic.minesweeper.utils import explored_mine, is_valid_action


class RewardFn(abc.ABC):
    @abc.abstractmethod
    def __call__(self, state: State, action: chex.Array) -> chex.Array:
        """Call method for computing the reward given current state and selected action."""


class DefaultRewardFn(RewardFn):
    """A dense reward function corresponding to the 3 possible events:
    - Revealing an empty square
    - Revealing a mine
    - Choosing an invalid action (an already revealed square)
    """

    def __init__(
        self,
        revealed_empty_square_reward: float,
        revealed_mine_reward: float,
        invalid_action_reward: float,
    ):
        self.revealed_empty_square_reward = revealed_empty_square_reward
        self.revelead_mine_reward = revealed_mine_reward
        self.invalid_action_reward = invalid_action_reward

    def __call__(self, state: State, action: chex.Array) -> chex.Array:
        return jnp.where(
            is_valid_action(state=state, action=action),
            jnp.where(
                explored_mine(state=state, action=action),
                jnp.array(self.revelead_mine_reward, float),
                jnp.array(self.revealed_empty_square_reward, float),
            ),
            jnp.array(self.invalid_action_reward, float),
        )
