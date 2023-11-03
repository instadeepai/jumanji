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

from jumanji.environments.logic.solitaire.types import Board
from jumanji.environments.logic.solitaire.utils import (
    all_possible_moves,
    move,
    possible_down_moves,
    possible_left_moves,
    possible_right_moves,
    possible_up_moves,
)


def test_down_moves(starting_board5x5: Board, board7x7: Board) -> None:
    assert jnp.array_equal(
        possible_down_moves(starting_board5x5),
        [
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    )

    assert jnp.array_equal(
        possible_down_moves(board7x7),
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
    )


def test_up_moves(starting_board5x5: Board, board7x7: Board) -> None:
    assert jnp.array_equal(
        possible_up_moves(starting_board5x5),
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ],
    )

    assert jnp.array_equal(
        possible_up_moves(board7x7),
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
    )


def test_left_moves(starting_board5x5: Board, board7x7: Board) -> None:
    assert jnp.array_equal(
        possible_left_moves(starting_board5x5),
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    )

    assert jnp.array_equal(
        possible_left_moves(board7x7),
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
    )


def test_right_moves(starting_board5x5: Board, board7x7: Board) -> None:
    assert jnp.array_equal(
        possible_right_moves(starting_board5x5),
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    )

    assert jnp.array_equal(
        possible_right_moves(board7x7),
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
    )


def test_move_up(starting_board5x5: Board, board7x7: Board) -> None:
    # 5x5 example
    moves = jnp.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ]
    )

    peg_position = jnp.argwhere(moves)[0]
    board, reward = move(starting_board5x5, (*peg_position, 0))
    assert jnp.array_equal(
        board,
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [0, 1, 0, 1, 0],
        ],
    )

    # 7x7 example
    possible_moves = possible_down_moves(board)
    peg_position = jnp.argwhere(possible_moves)[0]
    # Move peg at (2, 2) up.
    board, reward = move(board7x7, (2, 2, 0))
    assert jnp.array_equal(
        board,
        [
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
    )


def test_move_right(starting_board5x5: Board, board7x7: Board) -> None:
    # 5x5 example
    moves = jnp.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    peg_position = jnp.argwhere(moves)[0]
    board, reward = move(starting_board5x5, (*peg_position, 1))
    assert jnp.array_equal(
        board,
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
    )

    # 7x7 example
    possible_moves = possible_down_moves(board)
    peg_position = jnp.argwhere(possible_moves)[0]
    # Move peg at (2, 1) right.
    board, reward = move(board7x7, (2, 1, 1))
    assert jnp.array_equal(
        board,
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
    )


def test_move_down(starting_board5x5: Board, board7x7: Board) -> None:
    # 5x5 example
    moves = jnp.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    peg_position = jnp.argwhere(moves)[0]
    board, reward = move(starting_board5x5, (*peg_position, 2))
    assert jnp.array_equal(
        board,
        [
            [0, 1, 0, 1, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
    )

    # 7x7 example
    possible_moves = possible_down_moves(board)
    peg_position = jnp.argwhere(possible_moves)[0]
    board, reward = move(board7x7, (1, 2, 2))
    assert jnp.array_equal(
        board,
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
    )


def test_move_left(starting_board5x5: Board, board7x7: Board) -> None:
    # 5x5 example
    moves = jnp.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    peg_position = jnp.argwhere(moves)[0]
    board, reward = move(starting_board5x5, (*peg_position, 3))
    assert jnp.array_equal(
        board,
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
    )

    # 7x7 example
    possible_moves = possible_down_moves(board)
    peg_position = jnp.argwhere(possible_moves)[0]
    # Move peg at (2, 2) left.
    board, reward = move(board7x7, (2, 2, 3))
    assert jnp.array_equal(
        board,
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
    )


def test_all_possible_moves(starting_board5x5: Board, board7x7: Board) -> None:
    # Stack on last dimension to make it visually easier to compare.
    # Moves are in the order up, right, down, left.
    # Actions are row, col, direction.

    # 5x5 example
    assert jnp.array_equal(
        all_possible_moves(starting_board5x5),
        jnp.stack(
            (
                jnp.array(
                    [
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            0,
                            0,
                            1,
                            0,
                            0,
                        ],
                    ]
                ),
                jnp.array(
                    [
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            1,
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                    ]
                ),
                jnp.array(
                    [
                        [
                            0,
                            0,
                            1,
                            0,
                            0,
                        ],
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                    ]
                ),
                jnp.array(
                    [
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            0,
                            0,
                            0,
                            0,
                            1,
                        ],
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                    ]
                ),
            ),
            axis=-1,
        ),
    )

    # 7x7 example
    assert jnp.array_equal(
        all_possible_moves(board7x7),
        jnp.stack(
            (
                jnp.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                    ],
                ),
                jnp.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                    ],
                ),
                jnp.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                    ],
                ),
                jnp.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                    ],
                ),
            ),
            axis=-1,
        ),
    )


@pytest.mark.parametrize(
    "starting_position",
    [pytest.lazy_fixture("starting_board5x5"), pytest.lazy_fixture("board7x7")],
)
def test_move(starting_position: Board) -> None:
    moves = all_possible_moves(starting_position)
    # Check all possible moves.
    for action in jnp.argwhere(moves):
        board, reward = move(starting_position, action)
        difference = board ^ starting_position
        # Check that only 3 pegs have changed.
        # One peg is removed, one peg is moved (2 changes).
        assert difference.sum() == 3
        # One peg is removed.
        assert starting_position.sum() - board.sum() == 1
