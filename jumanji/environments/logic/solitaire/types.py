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
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

# Define a type alias for a board, which is an array.
Board: TypeAlias = chex.Array


@dataclass
class State:
    """
    board: the board, each element is a space for a possible peg. True corresponds
        to a peg and False corresponds to a hole. Even though the whole grid is stored,
        only a cross of width 3 and height 3 contains possible positions for storing pegs.
    step_count: the number of steps taken so far.
    action_mask: array of booleans that indicate the feasible actions, i.e. valid
        directions to move in.
    remaining: the number of pegs remaining on the board.
    key: random key used to generate random numbers at each step and for auto-reset.
    """

    board: Board  # (board_size, board_size)
    step_count: chex.Numeric  # ()
    action_mask: chex.Array  # (board_size, board_size, 4,)
    remaining: chex.Numeric  # ()
    key: chex.PRNGKey  # (2,)


class Observation(NamedTuple):
    """
    board: the board, each element is a space for a possible peg. True corresponds
        to a peg and False corresponds to a hole. Even though the whole grid is stored,
        only a cross of width 3 and height 3 contains possible positions for storing pegs.
    action_mask: array of booleans that indicate the feasible actions, i.e. valid
        directions to move in.
    """

    board: Board  # (board_size, board_size)
    action_mask: chex.Array  # (board_size, board_size, 4,)
