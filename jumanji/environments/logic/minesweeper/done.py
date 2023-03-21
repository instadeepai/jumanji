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

from jumanji.environments.logic.minesweeper.types import State
from jumanji.environments.logic.minesweeper.utils import (
    explored_mine,
    is_solved,
    is_valid_action,
)


class DoneFn(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, state: State, next_state: State, action: chex.Array
    ) -> chex.Array:
        """Call method for computing the done signal given the current and next state,
        and the action taken.
        """


class DefaultDoneFn(DoneFn):
    """Terminate the episode as soon as an invalid action is taken, a mine is explored,
    or the board is solved.
    """

    def __call__(
        self, state: State, next_state: State, action: chex.Array
    ) -> chex.Array:
        return (
            ~is_valid_action(state=state, action=action)
            | explored_mine(state=state, action=action)
            | is_solved(state=next_state)
        )
