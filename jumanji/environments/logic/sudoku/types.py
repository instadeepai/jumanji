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

from typing import TYPE_CHECKING, NamedTuple

import chex

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class State:
    """
    board: 2D-array of integers representing the board.
    action_mask: 3D-array of booleans that indicates valid actions.
    key: random key used for auto-reset.
    """

    board: chex.Array  # (board_size, board_size)
    action_mask: chex.Array  # (board_size, board_size, board_size)
    key: chex.PRNGKey  # # (2,).


class Observation(NamedTuple):
    """
    board: 2D-array of integers representing the board.
    action_mask: 3D-array of booleans that indicates valid actions.
    """

    board: chex.Array  # (board_size, board_size)
    action_mask: chex.Array  # (board_size, board_size, board_size)
