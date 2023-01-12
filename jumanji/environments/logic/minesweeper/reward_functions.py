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

from jumanji.environments.logic.minesweeper.constants import (
    REVEALED_EMPTY_SQUARE_REWARD,
    REVEALED_MINE_OR_INVALID_ACTION_REWARD,
)
from jumanji.environments.logic.minesweeper.types import State
from jumanji.environments.logic.minesweeper.utils import explored_mine, is_valid_action
from jumanji.types import Action


class RewardFunction(abc.ABC):
    @abc.abstractmethod
    def __call__(self, state: State, action: Action) -> chex.Array:
        """Call method for computing the reward given current state and selected action"""


class DefaultRewardFunction(RewardFunction):
    """A dense reward function: 1 for every timestep on which a mine is not explored
    (or a small penalty if action is invalid), otherwise 0"""

    def __call__(self, state: State, action: Action) -> chex.Array:
        return jnp.where(
            is_valid_action(state=state, action=action),
            jnp.where(
                explored_mine(state=state, action=action),
                jnp.float32(REVEALED_MINE_OR_INVALID_ACTION_REWARD),
                jnp.float32(REVEALED_EMPTY_SQUARE_REWARD),
            ),
            jnp.float32(REVEALED_MINE_OR_INVALID_ACTION_REWARD),
        )
