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

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.numpy import DeviceArray

from jumanji.environments.logic.game_2048.types import Board


def shift_nonzero_element(carry: Tuple) -> Tuple[DeviceArray, int]:
    """Shift nonzero element from index i to index j and increment j.
    For example, in the case of this column [2, 0, 2, 0], this method will be invoked
    when `i` equals 0 and 2, and it will return successively ([2, 0, 2, 0], `j` = 1)
    and ([2, 2, 2, 0], `j` = 2).

    Args:
        carry:
            col: a column of the board.
            i: the current index.
            j: the index of the nonzero element. It also represents the number of nonzero
            elements that have been shifted so far.

    Returns:
        A tuple containing the updated array (col) and the incremented target index (j).
    """
    col, j, i = carry
    col = col.at[j].set(col[i])
    j += 1
    return col, j


def shift_column_elements_up(carry: Tuple, i: int) -> Tuple[DeviceArray, None]:
    """This method calls `shift_nonzero_element` to shift non-zero elements in the column,
    and conducts the identity operation if the element is zero.

    Agrs:
        carry:
            col: a one-dimensional array representing a column of the board.
            j: the index of the non zero element. It also represents the number of non-zero
            elements that have been shifted so far.
        i: the current index.

    Returns:
        A tuple containing the updated column and None.
    """
    col, j = carry
    col, j = jax.lax.cond(
        col[i] != 0,
        shift_nonzero_element,
        lambda col_j_i: col_j_i[:2],
        (col, j, i),
    )
    return (col, j), None


def fill_with_zero(carry: Tuple[DeviceArray, int]) -> Tuple[DeviceArray, int]:
    """Fill the remaining elements of the column with zeros after shifting non-zero elements to the up.
    For example: if the initial column is [2, 0, 2, 0] then this method will be invoked when `j`
    equals to 2 and 3.

    Args:
        carry:
            col:  a column of the board.
            j: the index of the nonzero element. It also represents the number of nonzero
            elements that have been shifted so far.

    Returns:
        A tuple containing the updated column and incremented index.
    """
    col, j = carry
    col = col.at[j].set(0)
    j += 1
    return col, j


def shift_up(col: DeviceArray) -> DeviceArray:
    """Shift all the elements in a column up.
    For example: [2, 0, 2, 0] -> [2, 2, 0, 0]

    Args:
        col: a column of the board.

    Returns:
        The modified column with all the elements shifted up.
    """
    j = 0
    (col, j), _ = jax.lax.scan(  # In example: [2, 0, 2, 0] -> [2, 2, 2, 0]
        f=shift_column_elements_up, init=(col, j), xs=jnp.arange(len(col))
    )
    col, j = jax.lax.while_loop(  # In example: [2, 2, 2, 0] -> [2, 2, 0, 0]
        lambda col_j: col_j[1] < len(col_j[0]),
        fill_with_zero,
        (col, j),
    )
    return col


def merge_elements(carry: Tuple) -> Tuple[DeviceArray, float]:
    """Merge two adjacent elements in a column.
    For example: col = [1, 1, 2, 2] and i = 2 -> [1, 1, 3, 0], with a reward equal to 2³.

    Args:
        carry: a tuple containing the current state of the column, the current index,
        and the current reward.

    Returns:
        A tuple containing the modified column, and the updated reward.
    """
    col, reward, i = carry
    new_col_i = col[i] + 1
    col = col.at[i].set(new_col_i)
    col = col.at[i + 1].set(0)
    reward += 2**new_col_i
    return col, reward


def merge_equal_elements(
    carry: Tuple[DeviceArray, float], i: int
) -> Tuple[Tuple[DeviceArray, float], None]:
    """This function merges adjacent non-zero elements in the column of the board, if the
    two adjacent elements are equal.
    This function will examine each element individually to locate two adjacent equal elements.
    For example in the case of [1, 1, 2, 2], this method will call `merge_elements` for `i` equals
    to 0 and 2.

    Args:
        carry: a tuple containing the current state of the column, and the current reward.
        i: the current index.

    Returns:
        Tuple containing the updated column and the reward.
    """
    col, reward = carry
    col, reward = jax.lax.cond(
        ((col[i] != 0) & (col[i] == col[i + 1])),
        merge_elements,
        lambda col_reward_i: col_reward_i[:2],
        (col, reward, i),
    )
    return (col, reward), None


def merge_col(col: DeviceArray) -> Tuple[DeviceArray, float]:
    """Merge the elements of a column according to the rules of the 2048 game.
    For example: [0, 0, 2, 2] -> [0, 0, 3, 0] with a reward equal to 2³.

    Args:
        col: a column of the board.

    Returns:
        A tuple containing the modified column and the total reward obtained by
        merging the elements.
    """
    reward = 0.0
    elements_indices = jnp.arange(len(col) - 1)
    (col, reward), _ = jax.lax.scan(
        f=merge_equal_elements, init=(col, reward), xs=elements_indices
    )
    return col, reward


def move_up_col(
    carry: Tuple[Board, float], c: int, final_shift: bool = True
) -> Tuple[Tuple[Board, float], None]:
    """Move the elements in the specified column up and merge those that are equal in
    a single pass. `final_shift` is not needed when computing the action mask - this is
    because creating the action mask only requires knowledge of whether the board will
    have changed as a result of the action.

    For example: [2, 2, 1, 1] -> [3, 2, 0, 0].

    Args:
         carry: tuple containing the board and the additional reward.
         c: column index to perform the move and merge on.
         final_shift: is a flag to determine if the column should be shifted up once or
         twice. In the "get_action_mask" method, it is set to False, as the purpose is
         to check if the action is allowed and one shift is enough for this determination.

     Returns:
         Tuple containing the updated board and the additional reward.
    """
    board, additional_reward = carry
    col = board[:, c]
    col = shift_up(col)  # In example: [2, 2, 1, 1] -> [2, 2, 1, 1]
    col, reward = merge_col(col)  # In example: [2, 2, 1, 1] -> [3, 0, 2, 0]
    if final_shift:
        col = shift_up(col)  # In example: [3, 0, 2, 0] -> [3, 2, 0, 0]
    additional_reward += reward
    return (board.at[:, c].set(col), additional_reward), None


def move_up(board: Board, final_shift: bool = True) -> Tuple[Board, float]:
    """Move up."""
    additional_reward = 0.0
    col_indices = jnp.arange(board.shape[0])  # Board of size 4 -> [0, 1, 2, 3]
    (board, additional_reward), _ = jax.lax.scan(
        f=functools.partial(move_up_col, final_shift=final_shift),
        init=(board, additional_reward),
        xs=col_indices,
    )
    return board, additional_reward


def move_down(board: Board, final_shift: bool = True) -> Tuple[Board, float]:
    """Move down."""
    board, additional_reward = move_up(
        board=jnp.flip(board, 0), final_shift=final_shift
    )
    return jnp.flip(board, 0), additional_reward


def move_left(board: Board, final_shift: bool = True) -> Tuple[Board, float]:
    """Move left."""
    board, additional_reward = move_up(
        board=jnp.rot90(board, k=-1), final_shift=final_shift
    )
    return jnp.rot90(board, k=1), additional_reward


def move_right(board: Board, final_shift: bool = True) -> Tuple[Board, float]:
    """Move right."""
    board, additional_reward = move_up(
        board=jnp.rot90(board, k=1), final_shift=final_shift
    )
    return jnp.rot90(board, k=-1), additional_reward
