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
import pytest

from jumanji.environments.packing.flat_pack.reward import DenseReward, SparseReward
from jumanji.environments.packing.flat_pack.types import State


@pytest.fixture
def pieces() -> chex.Array:
    """An array containing 4 pieces."""

    return jnp.array(
        [
            [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 2.0, 2.0], [2.0, 2.0, 2.0], [0.0, 0.0, 2.0]],
            [[3.0, 0.0, 0.0], [3.0, 3.0, 0.0], [3.0, 3.0, 3.0]],
            [[4.0, 4.0, 0.0], [4.0, 4.0, 4.0], [0.0, 4.0, 4.0]],
        ],
        dtype=jnp.float32,
    )


@pytest.fixture()
def state_with_no_pieces_placed(
    solved_board: chex.Array, key: chex.PRNGKey, pieces: chex.Array
) -> State:
    """A board state with no pieces placed."""

    return State(
        row_nibs_idxs=jnp.array([2]),
        col_nibs_idxs=jnp.array([2]),
        num_pieces=4,
        pieces=pieces,
        solved_board=solved_board,
        action_mask=jnp.ones((4, 4, 2, 2), dtype=bool),
        placed_pieces=jnp.zeros(4, dtype=bool),
        current_board=jnp.zeros_like(solved_board),
        step_count=0,
        key=key,
    )


@pytest.fixture()
def state_with_piece_one_placed(
    solved_board: chex.Array,
    board_with_piece_one_placed: chex.Array,
    pieces: chex.Array,
    key: chex.PRNGKey,
) -> State:
    """A board state with piece one placed."""

    key, new_key = jax.random.split(key)
    return State(
        row_nibs_idxs=jnp.array([2]),
        col_nibs_idxs=jnp.array([2]),
        num_pieces=4,
        solved_board=solved_board,
        # TODO: Add the correct full action mask here.
        action_mask=jnp.array(
            [
                False,
                True,
                True,
                True,
            ]
        ),
        placed_pieces=jnp.array(
            [
                True,
                False,
                False,
                False,
            ]
        ),
        current_board=board_with_piece_one_placed,
        step_count=0,
        key=new_key,
        pieces=pieces,
    )


@pytest.fixture()
def state_needing_only_piece_one(
    solved_board: chex.Array,
    board_with_piece_one_placed: chex.Array,
    pieces: chex.Array,
    key: chex.PRNGKey,
) -> State:
    """A board state that one needs piece one to be fully completed."""

    key, new_key = jax.random.split(key)

    current_board = solved_board - board_with_piece_one_placed

    return State(
        row_nibs_idxs=jnp.array([2]),
        col_nibs_idxs=jnp.array([2]),
        num_pieces=4,
        solved_board=solved_board,
        # TODO: Add the correct full action mask here.
        action_mask=jnp.array(
            [
                True,
                False,
                True,
                True,
            ]
        ),
        placed_pieces=jnp.array(
            [
                True,
                False,
                False,
                False,
            ]
        ),
        current_board=current_board,
        step_count=3,
        pieces=pieces,
        key=new_key,
    )


@pytest.fixture()
def solved_state(
    solved_board: chex.Array,
    pieces: chex.Array,
    key: chex.PRNGKey,
) -> State:
    """A solved board state."""

    key, new_key = jax.random.split(key)

    return State(
        row_nibs_idxs=jnp.array([2]),
        col_nibs_idxs=jnp.array([2]),
        num_pieces=4,
        solved_board=solved_board,
        action_mask=jnp.ones((4, 4, 2, 2), dtype=bool),
        placed_pieces=jnp.array(
            [
                True,
                True,
                True,
                True,
            ]
        ),
        current_board=solved_board,
        step_count=4,
        pieces=pieces,
        key=new_key,
    )


@pytest.fixture()
def piece_one_misplaced(board_with_piece_one_placed: chex.Array) -> chex.Array:
    """A 2D array of zeros where piece one has been placed completely incorrectly.
    That is to say that there is no overlap between where the piece has been placed and
    where it should be placed to solve the puzzle."""

    # Shift all elements in the array two down and two to the right
    misplaced_piece = jnp.roll(board_with_piece_one_placed, shift=2, axis=0)
    misplaced_piece = jnp.roll(misplaced_piece, shift=2, axis=1)

    return misplaced_piece


def test_dense_reward(
    state_with_no_pieces_placed: State,
    state_with_piece_one_placed: State,
    piece_one_correctly_placed: chex.Array,
    piece_one_partially_placed: chex.Array,
    piece_one_misplaced: chex.Array,
) -> None:

    dense_reward = jax.jit(DenseReward())

    # Test placing piece one completely correctly
    reward = dense_reward(
        state=state_with_no_pieces_placed,
        action=piece_one_correctly_placed,
        is_valid=True,
        is_done=False,
        next_state=state_with_piece_one_placed,
    )
    assert reward == 6.0

    # Test placing piece one partially correct
    reward = dense_reward(
        state=state_with_no_pieces_placed,
        action=piece_one_partially_placed,
        is_valid=True,
        is_done=False,
        next_state=state_with_piece_one_placed,
    )
    assert reward == 2.0

    # Test placing a completely incorrect piece
    reward = dense_reward(
        state=state_with_no_pieces_placed,
        action=piece_one_misplaced,
        is_valid=True,
        is_done=False,
        next_state=state_with_piece_one_placed,
    )
    assert reward == 0.0

    # Test invalid action returns 0 reward.
    reward = dense_reward(
        state=state_with_no_pieces_placed,
        action=piece_one_correctly_placed,
        is_valid=False,
        is_done=False,
        next_state=state_with_piece_one_placed,
    )
    assert reward == 0.0


def test_sparse_reward(
    state_with_no_pieces_placed: State,
    state_with_piece_one_placed: State,
    solved_state: State,
    state_needing_only_piece_one: State,
    piece_one_correctly_placed: chex.Array,
) -> None:

    sparse_reward = jax.jit(SparseReward())

    # Test that a intermediate step returns 0 reward
    reward = sparse_reward(
        state=state_with_no_pieces_placed,
        action=piece_one_correctly_placed,
        next_state=state_with_piece_one_placed,
        is_valid=True,
        is_done=False,
    )
    assert reward == 0.0

    # Test that having `is_done` set to true does not automatically
    # give a reward of 1.
    reward = sparse_reward(
        state=state_with_no_pieces_placed,
        action=piece_one_correctly_placed,
        next_state=state_with_piece_one_placed,
        is_valid=True,
        is_done=True,
    )
    assert reward == 0.0

    # Test that a final correctly placed piece gives 1 reward.
    reward = sparse_reward(
        state=state_needing_only_piece_one,
        action=piece_one_correctly_placed,
        next_state=solved_state,
        is_valid=True,
        is_done=True,
    )
    assert reward == 1.0
