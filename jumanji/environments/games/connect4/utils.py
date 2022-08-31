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

from typing import Callable

from chex import Array
from jax import numpy as jnp

from jumanji.environments.games.connect4.constants import BOARD_HEIGHT, BOARD_WIDTH
from jumanji.environments.games.connect4.types import Board
from jumanji.types import Action


def get_highest_row(column: Array) -> Array:
    """Returns the highest row index with a token for a given column.

    Args:
        column: column of the board

    Returns:
        integer of the index of the highest non-empty row in the column.

    """
    return column.shape[0] - jnp.sum(jnp.abs(column))


def generate_pad_left(number_of_shifts: int) -> Callable[[Board], Board]:
    """Generate a padding left function (see the generated function's docstring
     for more info).
     A generator is used to avoid JIT recompilation.

    Args:
        number_of_shifts: number of shifts to apply.

    Returns:
        padding left function
    """

    def pad_left(board: Board) -> Board:
        """Shift the board to the right by padding number_of_shifts rows of zeroes at the top.

        Args:
            board: board to shift.

        Returns:
            the shifted board.

        """
        return jnp.pad(board, [(0, 0), (number_of_shifts, 0)])[:, :BOARD_WIDTH]

    return pad_left


pad_left_one = generate_pad_left(1)
pad_left_two = generate_pad_left(2)
pad_left_three = generate_pad_left(3)


def generate_pad_top(number_of_shifts: int) -> Callable[[Board], Board]:
    """Generate a padding top function (see the generated function's docstring
     for more info).
     A generator is used to avoid JIT recompilation.

    Args:
        number_of_shifts: number of shifts to apply.

    Returns:
        padding left function
    """

    def pad_top(board: Board) -> Board:
        """Shift the board to the bottom by padding number_of_shifts rows of zeroes at the top.

        Args:
            board: board to shift.

        Returns:
            the shifted board.

        """
        return jnp.pad(board, [(number_of_shifts, 0), (0, 0)])[:BOARD_HEIGHT, :]

    return pad_top


pad_top_one = generate_pad_top(1)
pad_top_two = generate_pad_top(2)
pad_top_three = generate_pad_top(3)


def generate_pad_diagonal_down(number_of_shifts: int) -> Callable[[Board], Board]:
    """Generate a padding diagonal down function (see the generated function's docstring
     for more info).
     A generator is used to avoid JIT recompilation.

    Args:
        number_of_shifts: number of shifts to apply.

    Returns:
        padding left function
    """

    def pad_diag_down(board: Board) -> Board:
        """Shift the board diagonally to the right and the bottom by padding
        number_of_shifts rows on the left and number_of_shifts rows at the top.

            Args:
                board: board to shift.

            Returns:
                the shifted board.

        """
        return jnp.pad(board, [(number_of_shifts, 0), (number_of_shifts, 0)])[
            :BOARD_HEIGHT, :BOARD_WIDTH
        ]

    return pad_diag_down


pad_diagonal_down_one = generate_pad_diagonal_down(1)
pad_diagonal_down_two = generate_pad_diagonal_down(2)
pad_diagonal_down_three = generate_pad_diagonal_down(3)


def generate_pad_diagonal_up(number_of_shifts: int) -> Callable[[Board], Board]:
    """Generate a padding diagonal up function (see the generated function's docstring
     for more info).
     A generator is used to avoid JIT recompilation.

    Args:
        number_of_shifts: number of shifts to apply.

    Returns:
        padding left function
    """

    def pad_diag_up(board: Board) -> Board:
        """Shift the board diagonally to the right and the top by padding
        number_of_shifts rows on the left and number_of_shifts rows at the bottom.

        Args:
            board: board to shift.

        Returns:
            the shifted board.

        """
        return jnp.pad(board, [(0, number_of_shifts), (number_of_shifts, 0)])[
            number_of_shifts:, :BOARD_WIDTH
        ]

    return pad_diag_up


pad_diagonal_up_one = generate_pad_diagonal_up(1)
pad_diagonal_up_two = generate_pad_diagonal_up(2)
pad_diagonal_up_three = generate_pad_diagonal_up(3)


def is_winning(board: Board) -> Array:
    """Checks if the board contains a winning position for the current player

    Args:
        board: board to search the position for

    Returns:
        array of booleans where True means the current player has won.

    """
    board = board == 1

    # horizontal check
    shifted_x = jnp.logical_and(board, pad_left_one(board))
    shifted_x = jnp.logical_and(shifted_x, pad_left_two(board))
    shifted_x = jnp.logical_and(shifted_x, pad_left_three(board))

    horizontal_check = jnp.sum(shifted_x)

    # vertical check
    shifted_x = jnp.logical_and(board, pad_top_one(board))
    shifted_x = jnp.logical_and(shifted_x, pad_top_two(board))
    shifted_x = jnp.logical_and(shifted_x, pad_top_three(board))

    vertical_check = jnp.sum(shifted_x)

    # diagonal check
    shifted_x = jnp.logical_and(board, pad_diagonal_down_one(board))
    shifted_x = jnp.logical_and(shifted_x, pad_diagonal_down_two(board))
    shifted_x = jnp.logical_and(shifted_x, pad_diagonal_down_three(board))

    diagonal_down_check = jnp.sum(shifted_x)

    # diagonal check
    shifted_x = jnp.logical_and(board, pad_diagonal_up_one(board))
    shifted_x = jnp.logical_and(shifted_x, pad_diagonal_up_two(board))
    shifted_x = jnp.logical_and(shifted_x, pad_diagonal_up_three(board))

    diagonal_up_check = jnp.sum(shifted_x)

    return horizontal_check + vertical_check + diagonal_down_check + diagonal_up_check


def update_board(board: Board, highest_row: Array, action: Action) -> Board:
    """Adds the token at the desired column and invert (multiply by -1) the board
    to show the board from the PoV of the next player.

    Note: action must be valid (i.e. highest_row > 0)

    Args:
         board: board to update,
         highest_row: highest row with a token in the action column,
         action: column to fill.

    Returns:
        the updated board from the PoV of the next player.
    """
    return -1 * board.at[highest_row - 1, action].set(1)


def board_full(board: Board) -> Array:
    """Checks if the board is full.

    Args:
        board: board to check.

    Returns:
        True if the board is full, False otherwise.

    """
    return jnp.count_nonzero(board[0, :]) == BOARD_WIDTH


def get_action_mask(board: Board) -> Array:
    """Returns a binary mask of the same length as the width of the board where 1 means a token
    can be inserted in the given column, 0 otherwise.

    Args:
        board: board to compute the action mask for

    Returns:
        binary action mask

    """
    return jnp.equal(board[0, :], 0).astype(bool)
