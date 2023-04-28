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
import jax.numpy as jnp
import pytest

from jumanji.environments.logic.sudoku.constants import BOARD_WIDTH
from jumanji.environments.logic.sudoku.utils import (
    puzzle_completed,
    update_action_mask,
    validate_board,
)

initial_board_sample = jnp.array(
    [
        [0, 0, 0, 8, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 4, 3],
        [5, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 7, 0, 8, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 2, 0, 0, 3, 0, 0, 0, 0],
        [6, 0, 0, 0, 0, 0, 0, 7, 5],
        [0, 0, 3, 4, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 6, 0, 0],
    ]
)

solved_board_sample = jnp.array(
    [
        [2, 3, 7, 8, 4, 1, 5, 6, 9],
        [1, 8, 6, 7, 9, 5, 2, 4, 3],
        [5, 9, 4, 3, 2, 6, 7, 1, 8],
        [3, 1, 5, 6, 7, 4, 8, 9, 2],
        [4, 6, 9, 5, 8, 2, 1, 3, 7],
        [7, 2, 8, 1, 3, 9, 4, 5, 6],
        [6, 4, 2, 9, 1, 8, 3, 7, 5],
        [8, 5, 3, 4, 6, 7, 9, 2, 1],
        [9, 7, 1, 2, 5, 3, 6, 8, 4],
    ]
)

empty_board = jnp.zeros((BOARD_WIDTH, BOARD_WIDTH), int)


@pytest.mark.parametrize(
    "board,expected_validation",
    [
        (solved_board_sample, True),
        (solved_board_sample.at[1, 2].set(0), False),
        (solved_board_sample.at[4, 7].set(4), False),
    ],
)
def test_puzzle_completed(board: chex.Array, expected_validation: chex.Array) -> None:
    """Tests that the puzzle_completed function returns the True for a solved
    board."""
    board = board - 1
    assert puzzle_completed(board) == expected_validation


@pytest.mark.parametrize(
    "board,expected_validation",
    [
        (solved_board_sample, True),
        (empty_board, True),
        (
            empty_board.at[0, 0].set(1).at[0, 4].set(1),
            False,
        ),
        (
            empty_board.at[0, 0].set(1).at[2, 2].set(1),
            False,
        ),
        (
            empty_board.at[0, 0].set(1).at[4, 0].set(1),
            False,
        ),
        (
            empty_board.at[0, 0].set(1).at[4, 0].set(2),
            True,
        ),
    ],
)
def test_validate_board(board: chex.Array, expected_validation: chex.Array) -> None:
    """Tests that the validate_board function returns True for a valid board."""

    board = board - 1
    assert validate_board(board) == expected_validation


@pytest.mark.parametrize("board", (initial_board_sample,))
def test_update_action_mask(board: chex.Array) -> None:
    """Tests that the update_action_mask function returns the correct action mask"""
    action_mask = board != 0

    board = jnp.array(board, dtype=jnp.int32) - 1
    action_mask = jnp.array(action_mask, dtype=jnp.int32)
    action_mask = 1 - jnp.expand_dims(action_mask, -1).repeat(BOARD_WIDTH, axis=-1)

    action_mask = update_action_mask(action_mask, board).astype(bool)

    for i in range(BOARD_WIDTH):
        for j in range(BOARD_WIDTH):
            for k in range(BOARD_WIDTH):
                action = [i, j, k]
                is_action_supposed_valid = action_mask[action[0], action[1], action[2]]
                new_board = board.at[action[0], action[1]].set(action[2])
                new_board_valid = validate_board(new_board)
                cell_empty = board[action[0], action[1]] == -1
                is_action_valid = new_board_valid and cell_empty
                assert is_action_supposed_valid == is_action_valid
