import chex
import jax.numpy as jnp
import pytest

from jumanji.environments.logic.sudoku.utils import validate_board


@pytest.mark.parametrize(
    "board,expected_validation",
    [
        (
            jnp.array(
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
            ),
            True,
        ),
        (
            jnp.array(
                [
                    [2, 3, 7, 8, 4, 1, 5, 6, 9],
                    [1, 8, 0, 7, 9, 5, 2, 4, 3],
                    [5, 9, 4, 3, 2, 6, 7, 1, 8],
                    [3, 1, 5, 6, 7, 4, 8, 9, 2],
                    [4, 6, 9, 5, 8, 2, 1, 3, 7],
                    [7, 2, 8, 1, 3, 9, 4, 5, 6],
                    [6, 4, 2, 9, 1, 8, 3, 7, 5],
                    [8, 5, 3, 4, 6, 7, 9, 2, 1],
                    [9, 7, 1, 2, 5, 3, 6, 8, 4],
                ]
            ),
            False,
        ),
        (
            jnp.array(
                [
                    [2, 3, 7, 8, 4, 1, 5, 6, 9],
                    [1, 8, 6, 7, 9, 5, 2, 4, 3],
                    [5, 9, 4, 3, 2, 6, 7, 1, 8],
                    [3, 1, 5, 6, 7, 4, 8, 9, 2],
                    [4, 6, 9, 5, 8, 2, 1, 4, 7],
                    [7, 2, 8, 1, 3, 9, 4, 5, 6],
                    [6, 4, 2, 9, 1, 8, 3, 7, 5],
                    [8, 5, 3, 4, 6, 7, 9, 2, 1],
                    [9, 7, 1, 2, 5, 3, 6, 8, 4],
                ]
            ),
            False,
        ),
    ],
)
def test_validate_board(board: chex.Array, expected_validation: chex.Array):
    """Tests that the validate_board function returns the expected return for a solved
    board."""
    board = board - 1
    assert validate_board(board) == expected_validation
