import pytest
from jax import numpy as jnp
from jax import random

from jumanji.connect4.constants import BOARD_HEIGHT, BOARD_WIDTH
from jumanji.connect4.utils import (
    board_full,
    get_highest_row,
    is_winning,
    pad_diagonal_down_one,
    pad_diagonal_up_one,
    pad_diagonal_up_three,
    pad_diagonal_up_two,
    pad_left_one,
    pad_left_three,
    pad_top_three,
    update_board,
)


@pytest.fixture
def single_token_board(empty_board: jnp.array) -> jnp.array:
    """Board with a single token at position [2, 2].

    _____________________________
    | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
    | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    -----------------------------

    """
    board = empty_board.at[2, 2].set(1)
    return board


@pytest.fixture
def full_column_board(empty_board: jnp.array) -> jnp.array:
    """Board where column 1 is full.

    _____________________________
    | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
    -----------------------------

    """
    board = empty_board.at[:, 1].set(jnp.ones_like(empty_board[:, 1]))
    return board


@pytest.mark.parametrize("column_index, expected_row_index", [(0, 6), (1, 0)])
def test_get_highest_row(
    full_column_board: jnp.array, column_index: int, expected_row_index: int
) -> None:
    """Tests get_highest_row and ensures the expected row indices are returned"""
    column = full_column_board[:, column_index]
    row_index = get_highest_row(column)

    assert row_index == expected_row_index


def test_padding_functions(
    single_token_board: jnp.array, empty_board: jnp.array
) -> None:
    """Tests that the padding functions are behaving as expected"""
    # board where just the bottom right corner is set to 1
    bottom_right_corner_board = empty_board.at[BOARD_HEIGHT - 1, BOARD_WIDTH - 1].set(1)

    board = pad_left_one(single_token_board)
    board = pad_left_three(board)
    board = pad_top_three(board)

    assert jnp.array_equal(board, bottom_right_corner_board)

    # board where just the top right corner is set to 1
    top_right_corner_board = empty_board.at[0, BOARD_WIDTH - 1].set(1)

    board = pad_diagonal_down_one(single_token_board)

    board_1 = pad_diagonal_up_one(board)
    board_1 = pad_diagonal_up_two(board_1)

    assert jnp.array_equal(board_1, top_right_corner_board)

    board_2 = pad_diagonal_up_three(board)

    assert jnp.array_equal(board_2, top_right_corner_board)


def test_is_winning_winning_boards(empty_board: jnp.array) -> None:
    """Tests is_winning on a combination of winning boards"""

    # winning column
    board = empty_board
    board = board.at[1, 1].set(1)
    board = board.at[2, 1].set(1)
    board = board.at[3, 1].set(1)
    board = board.at[4, 1].set(1)

    assert is_winning(board)

    # winning row
    board = empty_board
    board = board.at[0, 1].set(1)
    board = board.at[0, 2].set(1)
    board = board.at[0, 3].set(1)
    board = board.at[0, 4].set(1)

    assert is_winning(board)

    # winning first diagonal
    board = empty_board
    board = board.at[0, 0].set(1)
    board = board.at[1, 1].set(1)
    board = board.at[2, 2].set(1)
    board = board.at[3, 3].set(1)

    assert is_winning(board)

    # winning second diagonal
    board = empty_board
    board = board.at[0, 5].set(1)
    board = board.at[1, 4].set(1)
    board = board.at[2, 3].set(1)
    board = board.at[3, 2].set(1)

    assert is_winning(board)


def test_update_board(empty_board: jnp.array, full_column_board: jnp.array) -> None:
    """Test update_board by adding tokens to the second column until the board is full"""
    board = empty_board
    one = jnp.array(1)
    for i in reversed(range(1, BOARD_HEIGHT + 1)):
        board = update_board(board, i, one)

    assert jnp.array_equal(jnp.abs(board), full_column_board)


def test_board_full__full_board() -> None:
    """Test that board_full returns True given a full board"""
    key = random.PRNGKey(0)
    board = random.choice(key, jnp.array([-1, 1]), shape=(BOARD_HEIGHT, BOARD_WIDTH))
    assert board_full(board).all()


def test_board_full__empty_board(empty_board: jnp.array) -> None:
    """Test that board_full returns True given a full board"""
    assert ~board_full(empty_board).any()
