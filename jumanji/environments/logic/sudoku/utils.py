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

# def create_board_csv():
#     board = [
#         [0, 0, 0, 8, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 4, 3],
#         [5, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 7, 0, 8, 0, 0],
#         [0, 0, 0, 0, 0, 0, 1, 0, 0],
#         [0, 2, 0, 0, 3, 0, 0, 0, 0],
#         [6, 0, 0, 0, 0, 0, 0, 7, 5],
#         [0, 0, 3, 4, 0, 0, 0, 0, 0],
#         [0, 0, 0, 2, 0, 0, 6, 0, 0],
#     ]
#     np.savetxt("test_puzzle.csv", board, delimiter=",")

#     solved_board = [
#         [2, 3, 7, 8, 4, 1, 5, 6, 9],
#         [1, 8, 6, 7, 9, 5, 2, 4, 3],
#         [5, 9, 4, 3, 2, 6, 7, 1, 8],
#         [3, 1, 5, 6, 7, 4, 8, 9, 2],
#         [4, 6, 9, 5, 8, 2, 1, 3, 7],
#         [7, 2, 8, 1, 3, 9, 4, 5, 6],
#         [6, 4, 2, 9, 1, 8, 3, 7, 5],
#         [8, 5, 3, 4, 6, 7, 9, 2, 1],
#         [9, 7, 1, 2, 5, 3, 6, 8, 4],
#     ]

#     np.savetxt("test_solution.csv", solved_board, delimiter=",")


def apply_action(action: chex.Array, board: chex.Array) -> chex.Array:
    return board.at[action[0], action[1]].set(action[2])


def validate_board(board: chex.Array) -> chex.Array:
    """Checks that every row, column and 3x3 boxes includes all figures.

    Args:
        board: The sudoku board.

    Returns:
        condition: A `bool` indicator that validates a solution or rejects it.
    """

    def _validate_row(row: chex.Array) -> chex.Array:
        condition = jnp.sort(row) == jnp.arange(BOARD_WIDTH)
        return condition.all()

    condition_rows = jax.vmap(_validate_row)(board).all()
    condition_columns = jax.vmap(_validate_row)(board.T).all()
    condition_boxes = jax.vmap(_validate_row)(board[jnp.array(BOX_IDX)]).all()

    return condition_rows & condition_columns & condition_boxes


def update_action_mask(action_mask: chex.Array, board: chex.Array) -> chex.Array:
    """Updates the action mask according to the current board state.

    Args:
        action_mask: The action mask.
        board: The sudoku board.

    Returns:
        The updated action mask.
    """
    row_mask = 1 - jax.nn.one_hot(board, BOARD_WIDTH).any(axis=1) * 1
    column_mask = 1 - jax.nn.one_hot(board.T, BOARD_WIDTH).any(axis=1) * 1

    boxes = board.reshape(BOARD_WIDTH**2).take(jnp.array(BOX_IDX))
    box_mask = 1 - jax.nn.one_hot(boxes, BOARD_WIDTH).any(axis=1) * 1

    boxes_action_mask = action_mask.reshape(BOARD_WIDTH**2, BOARD_WIDTH)[
        jnp.array(BOX_IDX)
    ]
    boxes_action_mask *= box_mask.reshape(BOARD_WIDTH, 1, BOARD_WIDTH)

    action_mask = (
        action_mask.reshape(BOARD_WIDTH**2, BOARD_WIDTH)
        .at[jnp.array(BOX_IDX)]
        .set(boxes_action_mask)
        .reshape(BOARD_WIDTH, BOARD_WIDTH, BOARD_WIDTH)
    )
    action_mask *= row_mask.reshape(BOARD_WIDTH, 1, BOARD_WIDTH)
    action_mask *= column_mask.reshape(1, BOARD_WIDTH, BOARD_WIDTH)
    board_mask = (board == -1) * 1

    action_mask *= board_mask.reshape(BOARD_WIDTH, BOARD_WIDTH, 1)

    return action_mask
