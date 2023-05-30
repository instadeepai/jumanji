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

from itertools import product

import chex
import jax.numpy as jnp
import pytest

from jumanji.environments.logic.sudoku.constants import (
    BOARD_WIDTH,
    INITIAL_BOARD_SAMPLE,
    SOLVED_BOARD_SAMPLE,
)
from jumanji.environments.logic.sudoku.utils import (
    get_action_mask,
    puzzle_completed,
    validate_board,
)

EMPTY_BOARD = jnp.zeros((BOARD_WIDTH, BOARD_WIDTH), int)


@pytest.mark.parametrize(
    "board,expected_validation",
    [
        (SOLVED_BOARD_SAMPLE, True),
        (jnp.array(SOLVED_BOARD_SAMPLE).at[1, 2].set(0), False),
        (jnp.array(SOLVED_BOARD_SAMPLE).at[4, 7].set(4), False),
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
        (SOLVED_BOARD_SAMPLE, True),
        (EMPTY_BOARD, True),
        (
            EMPTY_BOARD.at[0, 0].set(1).at[0, 4].set(1),
            False,
        ),
        (
            EMPTY_BOARD.at[0, 0].set(1).at[2, 2].set(1),
            False,
        ),
        (
            EMPTY_BOARD.at[0, 0].set(1).at[4, 0].set(1),
            False,
        ),
        (
            EMPTY_BOARD.at[0, 0].set(1).at[4, 0].set(2),
            True,
        ),
    ],
)
def test_validate_board(board: chex.Array, expected_validation: chex.Array) -> None:
    """Tests that the validate_board function returns True for a valid board."""

    board = board - 1
    assert validate_board(board) == expected_validation


@pytest.mark.parametrize("board", (INITIAL_BOARD_SAMPLE,))
def test_get_action_mask(board: chex.Array) -> None:
    """Tests that the get_action_mask function returns the correct action mask"""

    board = jnp.array(board, dtype=jnp.int32) - 1

    action_mask = get_action_mask(board)

    for i, j, k in product(range(BOARD_WIDTH), range(BOARD_WIDTH), range(BOARD_WIDTH)):
        action = [i, j, k]
        is_action_supposed_valid = action_mask[action[0], action[1], action[2]]
        new_board = board.at[action[0], action[1]].set(action[2])
        new_board_valid = validate_board(new_board)
        cell_empty = board[action[0], action[1]] == -1
        is_action_valid = new_board_valid and cell_empty
        assert is_action_supposed_valid == is_action_valid
