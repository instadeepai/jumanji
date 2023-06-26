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

import jax.numpy as jnp
import pytest

from jumanji.environments.logic.game_2048.types import Board
from jumanji.environments.logic.game_2048.utils import (
    can_move_down,
    can_move_left,
    can_move_right,
    can_move_up,
    move_down,
    move_left,
    move_right,
    move_up,
)


@pytest.fixture
def board() -> Board:
    """Random board."""
    board = jnp.array([[1, 1, 0, 0], [0, 1, 1, 2], [1, 0, 0, 2], [2, 0, 0, 0]])
    return board


@pytest.fixture
def another_board() -> Board:
    """Random board."""
    board = jnp.array([[2, 1, 1, 0], [2, 3, 1, 2], [2, 3, 1, 2], [2, 3, 0, 0]])
    return board


@pytest.fixture
def board6x6() -> Board:
    """Random board with size 6x6."""
    board = jnp.array(
        [
            [2, 1, 1, 0, 0, 2],
            [2, 3, 1, 2, 1, 0],
            [2, 3, 1, 2, 3, 5],
            [2, 3, 0, 0, 3, 5],
            [2, 3, 0, 0, 0, 1],
            [2, 3, 0, 0, 1, 2],
        ]
    )
    return board


@pytest.fixture
def board8x8() -> Board:
    """Random board with size 8x8."""
    board = jnp.array(
        [
            [2, 1, 1, 0, 2, 1, 1, 0],
            [2, 3, 1, 2, 2, 1, 1, 0],
            [2, 3, 1, 2, 2, 1, 1, 0],
            [2, 3, 0, 0, 2, 1, 1, 0],
            [1, 1, 0, 0, 0, 1, 1, 2],
            [0, 1, 1, 2, 2, 3, 0, 0],
            [1, 0, 0, 2, 1, 2, 3, 5],
            [2, 0, 0, 0, 2, 1, 1, 5],
        ]
    )
    return board


def test_can_move_down(board: Board, another_board: Board) -> None:
    """Test checking if the board can move down."""
    assert can_move_down(board)
    assert can_move_down(another_board)
    board = jnp.array([[0, 0, 0, 0], [1, 0, 0, 0], [2, 1, 0, 0], [3, 2, 1, 0]])
    assert ~can_move_down(board)


def test_can_move_up(board: Board, another_board: Board) -> None:
    """Test checking if the board can move up."""
    assert can_move_up(board)
    assert can_move_up(another_board)
    board = jnp.array([[4, 2, 1, 0], [3, 1, 0, 0], [2, 0, 0, 0], [1, 0, 0, 0]])
    assert ~can_move_up(board)


def test_can_move_right(board: Board, another_board: Board) -> None:
    """Test checking if the board can move right."""
    assert can_move_right(board)
    assert can_move_right(another_board)
    board = jnp.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 2], [0, 1, 2, 3]])
    assert ~can_move_right(board)


def test_can_move_left(board: Board, another_board: Board) -> None:
    """Test checking if the board can move left."""
    assert can_move_left(board)
    assert can_move_left(another_board)
    board = jnp.array([[1, 2, 3, 4], [1, 2, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    assert ~can_move_left(board)


def test_move_down(board: Board, another_board: Board) -> None:
    """Test shifting the board cells down."""
    # First example.
    board, reward = move_down(board)
    expected_board = jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0], [2, 2, 1, 3]])
    assert jnp.array_equal(board, expected_board)
    assert reward == 16
    # Second example.
    board, reward = move_down(another_board)
    expected_board = jnp.array([[0, 0, 0, 0], [0, 1, 0, 0], [3, 3, 1, 0], [3, 4, 2, 3]])
    assert jnp.array_equal(board, expected_board)
    assert reward == 44


def test_move_up(board: Board, another_board: Board) -> None:
    """Test shifting the board cells up."""
    # First example.
    board, reward = move_up(board)
    expected_board = jnp.array([[2, 2, 1, 3], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert jnp.array_equal(board, expected_board)
    assert reward == 16
    # Second example.
    board, reward = move_up(another_board)
    expected_board = jnp.array([[3, 1, 2, 3], [3, 4, 1, 0], [0, 3, 0, 0], [0, 0, 0, 0]])
    assert jnp.array_equal(board, expected_board)
    assert reward == 44


def test_move_right(board: Board, another_board: Board) -> None:
    """Test shifting the board cells to the right."""
    # First example.
    board, reward = move_right(board)
    expected_board = jnp.array([[0, 0, 0, 2], [0, 0, 2, 2], [0, 0, 1, 2], [0, 0, 0, 2]])
    assert jnp.array_equal(board, expected_board)
    assert reward == 8
    # Second example.
    board, reward = move_right(another_board)
    expected_board = jnp.array([[0, 0, 2, 2], [2, 3, 1, 2], [2, 3, 1, 2], [0, 0, 2, 3]])
    assert jnp.array_equal(board, expected_board)
    assert reward == 4


def test_move_left(board: Board, another_board: Board) -> None:
    """Test shifting the board cells to the left."""
    # First example.
    board, reward = move_left(board)
    expected_board = jnp.array([[2, 0, 0, 0], [2, 2, 0, 0], [1, 2, 0, 0], [2, 0, 0, 0]])
    assert jnp.array_equal(board, expected_board)
    assert reward == 8
    # Second example.
    board, reward = move_left(another_board)
    expected_board = jnp.array([[2, 2, 0, 0], [2, 3, 1, 2], [2, 3, 1, 2], [2, 3, 0, 0]])
    assert jnp.array_equal(board, expected_board)
    assert reward == 4


def test_board_size_6(board6x6: Board) -> None:
    """Validate that various actions can be performed on a 6x6 game board."""
    board_up, reward = move_up(board6x6)
    expected_board = jnp.array(
        [
            [3, 1, 2, 3, 1, 2],
            [3, 4, 1, 0, 4, 6],
            [3, 4, 0, 0, 1, 1],
            [0, 3, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    assert jnp.array_equal(expected_board, board_up)
    assert reward == 148

    board_down, reward = move_down(board6x6)
    expected_board = jnp.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 2],
            [3, 3, 0, 0, 1, 6],
            [3, 4, 1, 0, 4, 1],
            [3, 4, 2, 3, 1, 2],
        ]
    )
    assert jnp.array_equal(expected_board, board_down)
    assert reward == 148

    board_left, reward = move_left(board6x6)
    expected_board = jnp.array(
        [
            [2, 2, 2, 0, 0, 0],
            [2, 3, 1, 2, 1, 0],
            [2, 3, 1, 2, 3, 5],
            [2, 4, 5, 0, 0, 0],
            [2, 3, 1, 0, 0, 0],
            [2, 3, 1, 2, 0, 0],
        ]
    )
    assert jnp.array_equal(expected_board, board_left)
    assert reward == 20

    board_right, reward = move_right(board6x6)
    expected_board = jnp.array(
        [
            [0, 0, 0, 2, 2, 2],
            [0, 2, 3, 1, 2, 1],
            [2, 3, 1, 2, 3, 5],
            [0, 0, 0, 2, 4, 5],
            [0, 0, 0, 2, 3, 1],
            [0, 0, 2, 3, 1, 2],
        ]
    )
    assert jnp.array_equal(expected_board, board_right)
    assert reward == 20


def test_board_size_8(board8x8: Board) -> None:
    """Validate that various actions can be performed on a 8x8 game board."""
    board_up, reward = move_up(board8x8)
    expected_board = jnp.array(
        [
            [3, 1, 2, 3, 3, 2, 2, 2],
            [3, 4, 2, 3, 3, 2, 2, 6],
            [2, 3, 0, 0, 2, 1, 1, 0],
            [2, 2, 0, 0, 1, 3, 3, 0],
            [0, 0, 0, 0, 2, 2, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    assert jnp.array_equal(expected_board, board_up)
    assert reward == 160

    board_down, reward = move_down(board8x8)
    expected_board = jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 2, 2, 1, 0],
            [3, 1, 0, 0, 3, 2, 2, 0],
            [3, 3, 0, 0, 3, 3, 2, 0],
            [2, 4, 2, 3, 1, 2, 3, 2],
            [2, 2, 2, 3, 2, 1, 1, 6],
        ]
    )
    assert jnp.array_equal(expected_board, board_down)
    assert reward == 160

    board_left, reward = move_left(board8x8)
    expected_board = jnp.array(
        [
            [2, 2, 2, 2, 0, 0, 0, 0],
            [2, 3, 1, 3, 2, 0, 0, 0],
            [2, 3, 1, 3, 2, 0, 0, 0],
            [2, 3, 2, 2, 0, 0, 0, 0],
            [2, 2, 2, 0, 0, 0, 0, 0],
            [2, 3, 3, 0, 0, 0, 0, 0],
            [1, 2, 1, 2, 3, 5, 0, 0],
            [3, 2, 5, 0, 0, 0, 0, 0],
        ]
    )
    assert jnp.array_equal(expected_board, board_left)
    assert reward == 68

    board_right, reward = move_right(board8x8)
    expected_board = jnp.array(
        [
            [0, 0, 0, 0, 2, 2, 2, 2],
            [0, 0, 0, 2, 3, 1, 3, 2],
            [0, 0, 0, 2, 3, 1, 3, 2],
            [0, 0, 0, 0, 2, 3, 2, 2],
            [0, 0, 0, 0, 0, 2, 2, 2],
            [0, 0, 0, 0, 0, 2, 3, 3],
            [0, 0, 1, 2, 1, 2, 3, 5],
            [0, 0, 0, 0, 0, 3, 2, 5],
        ]
    )
    assert jnp.array_equal(expected_board, board_right)
    assert reward == 68
