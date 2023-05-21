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

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass


class Observation(NamedTuple):
    """
    current_board: 2D array with the current state of board.
    pieces: 3D array with the pieces to be placed on the board. Here each piece is a
        2D array with shape (3, 3).
    board_action_mask: 2D array showing where on the board pieces have already been
        placed.
    piece_action_mask: array showing which pieces can be placed on the board.
    """

    current_board: chex.Array  # (num_rows, num_cols)
    pieces: chex.Array  # (num_pieces, 3, 3)
    board_action_mask: chex.Array  # (num_rows, num_cols)
    piece_action_mask: chex.Array  # (num_pieces,)


@dataclass
class State:
    """
    row_nibs_idxs: array containing row indices for selecting piece nibs.
        it will be of length num_sig_rows where sig refers to significant implying
        that this row will contain puzzles nibs.
    col_nibs_idxs: array containing column indices for selecting piece nibs.
        it will be of length num_sig_cols where sig refers to significant implying
        that this column will contain puzzles nibs.
    num_pieces: number of pieces in the jigsaw puzzle.
    solved_board: 2D array showing the solved board state.
    pieces: 3D array with the pieces to be placed on the board. Here each piece is a
        2D array with shape (3, 3).
    board_action_mask: 2D array showing where pieces on the board have been
        placed.
    piece_action_mask: array showing which pieces can be placed on the board.
    current_board: 2D array with the current state of board.
    step_count: number of steps taken in the environment.
    key: random key used for board generation.
    """

    row_nibs_idxs: chex.Array  # (num_sig_rows,)
    col_nibs_idxs: chex.Array  # (num_sig_cols,)
    num_pieces: chex.Numeric  # ()
    solved_board: chex.Array  # (num_rows, num_cols)
    pieces: chex.Array  # (num_pieces, 3, 3)
    board_action_mask: chex.Array  # (num_rows, num_cols)
    piece_action_mask: chex.Array  # (num_pieces,)
    current_board: chex.Array  # (num_rows, num_cols)
    step_count: chex.Numeric  # ()
    key: chex.PRNGKey  # (2,)
