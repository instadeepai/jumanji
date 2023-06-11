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


def shift_row_elements_left(origin: int, carry: Tuple) -> Tuple[DeviceArray, int]:
    """This method shifts non-zero elements in the row, and conducts the identity operation if the
    element is zero.

    Agrs:
        origin: the index to shift from.
        carry:
            row: a one-dimensional array representing a row of the board.
            target: the index to shift from. It also represents the number of non-zero elements
            that have been shifted so far.

    Returns:
        A tuple containing the updated row and the new target.
    """
    row, target = carry
    new_row_target, new_target = jax.lax.cond(
        row[origin] != 0,
        lambda row, target, origin: (row[origin], target + 1),
        lambda row, target, origin: (row[target], target),
        row,
        target,
        origin,
    )
    row = row.at[target].set(new_row_target)
    return row, new_target


def fill_with_zero(target: int, row: chex.Array) -> chex.Array:
    """Fill the remaining elements of the row with zeros after shifting non-zero elements to the left.
    For example: if the initial row is [2, 0, 2, 0] then this method will be invoked when `target`
    equals to 2 and 3.

    Args:
        target: index to fill with 0.
        row: a one-dimensional array representing a row of the board.

    Returns:
        The updated row.
    """
    row = row.at[target].set(0)
    return row


def shift_left(row: DeviceArray) -> DeviceArray:
    """Shift all the elements in a row left.
    For example: [2, 0, 2, 0] -> [2, 2, 0, 0]

    Args:
        row: a one-dimensional array representing a row of the board.

    Returns:
        The modified row with all the elements shifted left.
    """
    target = 0
    row, target = jax.lax.fori_loop(
        0, row.shape[0], shift_row_elements_left, (row, target)
    )
    row = jax.lax.fori_loop(target, row.shape[0], fill_with_zero, row)
    return row


def merge_equal_elements(
    target: int, carry: Tuple[DeviceArray, float]
) -> Tuple[DeviceArray, float]:
    """This function merges adjacent non-zero elements in the row of the board, if the
    two adjacent elements are equal.
    This function will examine each element individually to locate two adjacent equal elements.
    For example in the case of [1, 1, 2, 2], this method will merge elements for `target` equals
    to 0 and 2.

    Args:
        target: index to merge into.
        carry: a tuple containing the current state of the row and the current reward.

    Returns:
        Tuple containing the updated row and the reward.
    """
    row, reward = carry
    new_row_target, new_row_target_plus_1, additional_reward = jax.lax.cond(
        (row[target] != 0) & (row[target] == row[target + 1]),
        lambda row, target: (row[target] + 1, 0, 2 ** (row[target] + 1)),
        lambda row, target: (row[target], row[target + 1], 0),
        row,
        target,
    )
    row = row.at[target].set(new_row_target)
    row = row.at[target + 1].set(new_row_target_plus_1)
    reward += additional_reward
    return row, reward


def merge_row(row: DeviceArray) -> Tuple[DeviceArray, float]:
    """Merge the elements of a row according to the rules of the 2048 game.
    For example: [0, 0, 2, 2] -> [0, 0, 3, 0] with a reward equal to 2³.

    Args:
        row: a one-dimensional array representing a row of the board.

    Returns:
        A tuple containing the modified row and the total reward obtained by
        merging the elements.
    """
    reward = 0.0
    row, reward = jax.lax.fori_loop(
        0, row.shape[0] - 1, merge_equal_elements, (row, reward)
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
         row: a one-dimensional array representing a row of the board.
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
