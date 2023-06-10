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
from jax.numpy import DeviceArray

from jumanji.environments.logic.game_2048.types import Board


def shift_row_elements_left(carry: Tuple, i: int) -> Tuple[DeviceArray, None]:
    """This method shifts non-zero elements in the row, and conducts the identity operation if the
    element is zero.

    Agrs:
        carry:
            row: a one-dimensional array representing a row of the board.
            j: the index of the non zero element. It also represents the number of non-zero
            elements that have been shifted so far.
        i: the current index.

    Returns:
        A tuple containing the updated row and None.
    """
    row, j = carry
    new_row_j, new_j = jax.lax.cond(
        row[i] != 0,
        lambda row, j, i: (row[i], j + 1),
        lambda row, j, i: (row[j], j),
        row,
        j,
        i,
    )
    row = row.at[j].set(new_row_j)
    return (row, new_j), None


def fill_with_zero(carry: Tuple[DeviceArray, int]) -> Tuple[DeviceArray, int]:
    """Fill the remaining elements of the row with zeros after shifting non-zero elements to the left.
    For example: if the initial row is [2, 0, 2, 0] then this method will be invoked when `j`
    equals to 2 and 3.

    Args:
        carry:
            row:  a row of the board.
            j: the index of the nonzero element. It also represents the number of nonzero
            elements that have been shifted so far.

    Returns:
        A tuple containing the updated row and incremented index.
    """
    row, j = carry
    row = row.at[j].set(0)
    j += 1
    return row, j


def shift_left(row: DeviceArray) -> DeviceArray:
    """Shift all the elements in a row left.
    For example: [2, 0, 2, 0] -> [2, 2, 0, 0]

    Args:
        row: a row of the board.

    Returns:
        The modified row with all the elements shifted left.
    """
    j = 0
    (row, j), _ = jax.lax.scan(  # In example: [2, 0, 2, 0] -> [2, 2, 2, 0]
        f=shift_row_elements_left, init=(row, j), xs=jnp.arange(len(row))
    )
    row, j = jax.lax.while_loop(  # In example: [2, 2, 2, 0] -> [2, 2, 0, 0]
        lambda row_j: row_j[1] < len(row_j[0]),
        fill_with_zero,
        (row, j),
    )
    return row


def merge_equal_elements(
    carry: Tuple[DeviceArray, float], i: int
) -> Tuple[Tuple[DeviceArray, float], None]:
    """This function merges adjacent non-zero elements in the row of the board, if the
    two adjacent elements are equal.
    This function will examine each element individually to locate two adjacent equal elements.
    For example in the case of [1, 1, 2, 2], this method will merge elements for `i` equals
    to 0 and 2.

    Args:
        carry: a tuple containing the current state of the row, and the current reward.
        i: the current index.

    Returns:
        Tuple containing the updated row and the reward.
    """
    row, reward = carry
    new_row_i, new_row_i_plus_1, additional_reward = jax.lax.cond(
        (row[i] != 0) & (row[i] == row[i + 1]),
        lambda row, i: (row[i] + 1, 0, 2 ** (row[i] + 1)),
        lambda row, i: (row[i], row[i + 1], 0),
        row,
        i,
    )
    row = row.at[i].set(new_row_i)
    row = row.at[i + 1].set(new_row_i_plus_1)
    reward += additional_reward
    return (row, reward), None


def merge_row(row: DeviceArray) -> Tuple[DeviceArray, float]:
    """Merge the elements of a row according to the rules of the 2048 game.
    For example: [0, 0, 2, 2] -> [0, 0, 3, 0] with a reward equal to 2³.

    Args:
        row: a row of the board.

    Returns:
        A tuple containing the modified row and the total reward obtained by
        merging the elements.
    """
    reward = 0.0
    elements_indices = jnp.arange(len(row) - 1)
    (row, reward), _ = jax.lax.scan(
        f=merge_equal_elements, init=(row, reward), xs=elements_indices
    )
    return row, reward


def move_left_row(
    row: chex.Array, final_shift: bool = True
) -> Tuple[chex.Array, float]:
    """Move the elements in the specified row left and merge those that are equal in
    a single pass. `final_shift` is not needed when computing the action mask - this is
    because creating the action mask only requires knowledge of whether the board will
    have changed as a result of the action.

    For example: [2, 2, 1, 1] -> [3, 2, 0, 0].

    Args:
         row: a row of the board.
         final_shift: is a flag to determine if the row should be shifted left once or
         twice. In the "get_action_mask" method, it is set to False, as the purpose is
         to check if the action is allowed and one shift is enough for this determination.

     Returns:
         Tuple containing the updated board and the additional reward.
    """
    row = shift_left(row)  # In example: [2, 2, 1, 1] -> [2, 2, 1, 1]
    row, reward = merge_row(row)  # In example: [2, 2, 1, 1] -> [3, 0, 2, 0]
    if final_shift:
        row = shift_left(row)  # In example: [3, 0, 2, 0] -> [3, 2, 0, 0]
    return row, reward


def move_left(board: Board, final_shift: bool = True) -> Tuple[Board, float]:
    """Move left."""
    board, additional_reward = jax.vmap(move_left_row, (0, None))(board, final_shift)
    return board, additional_reward.sum()


def transform_board(board: Board, action: int) -> Board:
    """Transform board."""
    return jax.lax.switch(
        action,
        [
            lambda board: jnp.transpose(board),
            lambda board: jnp.flip(board, 1),
            lambda board: jnp.flip(jnp.transpose(board)),
            lambda board: board,
        ],
        board,
    )


def move(board: Board, action: int, final_shift: bool = True) -> Tuple[Board, float]:
    """Move."""
    board = transform_board(board, action)
    board, additional_reward = move_left(board, final_shift)
    board = transform_board(board, action)
    return board, additional_reward


def move_up(board: Board, final_shift: bool = True) -> Tuple[Board, float]:
    """Move up."""
    return move(board=board, action=0, final_shift=final_shift)


def move_right(board: Board, final_shift: bool = True) -> Tuple[Board, float]:
    """Move right."""
    return move(board=board, action=1, final_shift=final_shift)


def move_down(board: Board, final_shift: bool = True) -> Tuple[Board, float]:
    """Move down."""
    return move(board=board, action=2, final_shift=final_shift)
