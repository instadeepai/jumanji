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

from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.logic.solitaire.types import Board


def direction_int_to_tuple(action: int) -> chex.Array:
    """Convert action to direction."""
    return jax.lax.switch(
        action,
        [
            lambda: jnp.array((-1, 0)),
            lambda: jnp.array((0, 1)),
            lambda: jnp.array((1, 0)),
            lambda: jnp.array((0, -1)),
        ],
    )


def playing_board(board_size: int) -> Board:
    """Create a playing board of a given size."""
    mid_size = board_size // 2
    # Create empty board
    board = jnp.zeros((board_size, board_size), dtype=bool)

    # Fill in all the pegs.
    # This forms a 3x3 cross across the middle of the board.
    board = board.at[:, mid_size - 1 : mid_size + 2].set(True)
    board = board.at[mid_size - 1 : mid_size + 2, :].set(True)

    return board


def possible_moves(board: Board, action: int) -> Board:
    """Check which pegs can be moved left."""
    # A move involves jumping a peg over another peg into a hole.
    board_mask = playing_board(len(board))
    direction = direction_int_to_tuple(action)
    pegs_to_play = board
    pegs_to_remove = jnp.roll(board & board_mask, -direction, axis=(0, 1))
    holes_to_fill = jnp.roll(~board & board_mask, -direction * 2, axis=(0, 1))

    # Check if the pegs can be moved.
    possible_moves = pegs_to_play & pegs_to_remove & holes_to_fill

    # Mask out moves that wrap around the board.
    board = jax.lax.switch(
        action,
        [
            lambda: board.at[:2, :].set(False),
            lambda: board.at[:, -2:].set(False),
            lambda: board.at[-2:, :].set(False),
            lambda: board.at[:, :2].set(False),
        ],
    )

    return possible_moves


def possible_up_moves(board: Board) -> Board:
    return possible_moves(board, 0)


def possible_right_moves(board: Board) -> Board:
    return possible_moves(board, 1)


def possible_down_moves(board: Board) -> Board:
    return possible_moves(board, 2)


def possible_left_moves(board: Board) -> Board:
    return possible_moves(board, 3)


def all_possible_moves(board: Board) -> Board:
    """Check which pegs can be moved in any direction."""
    # Check all directions.
    moves = jnp.stack(
        jax.vmap(possible_moves, in_axes=(None, 0))(board, jnp.arange(4)),
        axis=-1,
    )
    return moves


def move(board: Board, action: Tuple) -> Tuple[Board, float]:
    """Move the board with action."""
    # Action is a tuple of (row, col, direction).
    row, col, direction = action
    # Convert integer to a useful direction.
    direction = direction_int_to_tuple(direction)
    # Remove current peg.
    board = board.at[row, col].set(False)
    # Remove jumped-over peg.
    board = board.at[row + direction[0], col + direction[1]].set(False)
    # Add new peg.
    board = board.at[row + direction[0] * 2, col + direction[1] * 2].set(True)
    # Reward is always 1.
    return board, 1.0
