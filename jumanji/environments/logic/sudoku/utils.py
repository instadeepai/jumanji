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

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.logic.sudoku.constants import BOARD_WIDTH, BOX_IDX


def apply_action(action: chex.Array, board: chex.Array) -> chex.Array:
    return board.at[action[0], action[1]].set(action[2])


def is_puzzle_solved(board: chex.Array) -> chex.Array:
    """Checks that every row, column and 3x3 boxes includes all figures.

    Args:
        board: The sudoku board.

    Returns:
        condition: A `bool` indicator that validates if the puzzle is solved.
    """

    def _validate_row(row: chex.Array) -> chex.Array:
        condition = jnp.sort(row) == jnp.arange(BOARD_WIDTH)
        return condition.all()

    condition_rows = jax.vmap(_validate_row)(board).all()
    condition_columns = jax.vmap(_validate_row)(board.T).all()
    condition_boxes = jax.vmap(_validate_row)(
        jnp.take(board, jnp.asarray(BOX_IDX))
    ).all()
    return condition_rows & condition_columns & condition_boxes


def validate_board(board: chex.Array) -> chex.Array:
    """Checks that every row, column and 3x3 boxes contain no duplicate number.

    Args:
        board: The sudoku board.

    Returns:
        condition: A `bool` indicator that validates a board or not.
    """

    def _validate_row(row: chex.Array) -> chex.Array:
        return jax.nn.one_hot(row, BOARD_WIDTH).sum(axis=0).max() <= 1

    condition_rows = jax.vmap(_validate_row)(board).all()
    condition_columns = jax.vmap(_validate_row)(board.T).all()
    condition_boxes = jax.vmap(_validate_row)(jnp.take(board, BOX_IDX)).all()

    return condition_rows & condition_columns & condition_boxes


def get_action_mask(board: chex.Array) -> chex.Array:
    """Updates the action mask according to the current board state.

    Args:
        board: The sudoku board.

    Returns:
        The updated action mask.
    """
    action_mask = board == -1
    action_mask = jnp.expand_dims(action_mask, -1).repeat(BOARD_WIDTH, axis=-1)

    row_mask = ~jax.nn.one_hot(board, BOARD_WIDTH).any(axis=1)
    column_mask = ~jax.nn.one_hot(board.T, BOARD_WIDTH).any(axis=1)

    boxes = board.reshape(BOARD_WIDTH**2).take(BOX_IDX)
    box_mask = ~jax.nn.one_hot(boxes, BOARD_WIDTH).any(axis=1)

    boxes_action_mask = action_mask.reshape(BOARD_WIDTH**2, BOARD_WIDTH)[BOX_IDX]
    boxes_action_mask *= box_mask.reshape(BOARD_WIDTH, 1, BOARD_WIDTH)

    action_mask = (
        action_mask.reshape(BOARD_WIDTH**2, BOARD_WIDTH)
        .at[BOX_IDX]
        .set(boxes_action_mask)
        .reshape(BOARD_WIDTH, BOARD_WIDTH, BOARD_WIDTH)
    )
    action_mask &= row_mask.reshape(BOARD_WIDTH, 1, BOARD_WIDTH)
    action_mask &= column_mask.reshape(1, BOARD_WIDTH, BOARD_WIDTH)

    return action_mask
