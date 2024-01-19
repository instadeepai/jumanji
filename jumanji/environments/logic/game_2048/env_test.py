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

from jumanji.environments.logic.game_2048.env import Game2048
from jumanji.environments.logic.game_2048.types import Board, State
from jumanji.testing.env_not_smoke import (
    check_env_does_not_smoke,
    check_env_specs_does_not_smoke,
)
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import TimeStep


@pytest.fixture
def game_2048() -> Game2048:
    """Instantiates a default Game2048 environment."""
    return Game2048()


@pytest.fixture
def board() -> Board:
    """Random board."""
    board = jnp.array([[1, 1, 2, 2], [3, 4, 0, 0], [0, 2, 0, 0], [0, 5, 0, 0]])
    return board


def test_game_2048__reset_jit(game_2048: Game2048) -> None:
    """Confirm that the reset method is only compiled once when jitted."""
    chex.clear_trace_counter()
    reset_fn = jax.jit(chex.assert_max_traces(game_2048.reset, n=1))
    key = jax.random.PRNGKey(0)
    state, timestep = reset_fn(key)

    # Verify the data type of the output.
    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)

    # Check that the state is made of DeviceArrays, this is false for the non-jitted.
    assert_is_jax_array_tree(state)

    # Call again to check it does not compile twice.
    state, timestep = reset_fn(key)
    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)


def test_game_2048__step_jit(game_2048: Game2048) -> None:
    """Confirm that the step is only compiled once when jitted."""
    key = jax.random.PRNGKey(0)
    state, timestep = game_2048.reset(key)
    action = jnp.argmax(state.action_mask)

    chex.clear_trace_counter()
    step_fn = jax.jit(chex.assert_max_traces(game_2048.step, n=1))

    new_state, next_timestep = step_fn(state, action)
    # Check that the state has changed.
    assert not jnp.array_equal(new_state.board, state.board)

    # Check that the state is made of DeviceArrays, this is false for the non-jitted.
    assert_is_jax_array_tree(new_state)

    # New step
    state = new_state
    action = jnp.argmax(state.action_mask)
    new_state, next_timestep = step_fn(state, action)

    # Check that the state has changed
    assert not jnp.array_equal(new_state.board, state.board)


def test_game_2048__step_invalid(game_2048: Game2048) -> None:
    """Confirm that performing step on an invalid action does nothing."""
    state = State(
        board=jnp.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        step_count=jnp.array(0),
        action_mask=jnp.array([False, True, True, True]),
        score=jnp.array(0),
        key=jax.random.PRNGKey(0),
    )
    action = jnp.array(0)
    step_fn = jax.jit(game_2048.step)
    new_state, next_timestep = step_fn(state, action)
    assert jnp.array_equal(state.board, new_state.board)
    assert jnp.array_equal(state.step_count + 1, new_state.step_count)
    assert jnp.array_equal(state.action_mask, new_state.action_mask)
    assert jnp.array_equal(state.score, new_state.score)


def test_game_2048__step_action_mask(game_2048: Game2048) -> None:
    """Verify that the action mask returned from `step` is correct."""
    state = State(
        board=jnp.array([[0, 1, 2, 3], [3, 1, 2, 3], [1, 2, 3, 4], [4, 3, 2, 1]]),
        step_count=jnp.array(0),
        action_mask=jnp.array([True, False, True, True]),
        score=jnp.array(0),
        key=jax.random.PRNGKey(0),
    )
    action = jnp.array(3)
    step_fn = jax.jit(game_2048.step)
    new_state, next_timestep = step_fn(state, action)
    expected_action_mask = jnp.array([False, False, False, False])
    assert jnp.array_equal(new_state.action_mask, expected_action_mask)


def test_game_2048__generate_board(game_2048: Game2048) -> None:
    """Confirm that `generate_board` method creates an initial board that
    follows the rules of the 2048 game."""
    key = jax.random.PRNGKey(0)
    generate_board = jax.jit(game_2048._generate_board)
    board = generate_board(key)
    # Check that all the tiles are zero except one tile with value (1 or 2).
    assert jax.numpy.count_nonzero(board) == 1
    new_value_is_one = jnp.sum(board == 1) == 1
    new_value_is_two = jnp.sum(board == 2) == 1
    assert new_value_is_one ^ new_value_is_two


def test_game_2048__add_random_cell(game_2048: Game2048, board: Board) -> None:
    """Validate that add_random_cell places a 1 or 2 in an empty spot on the board."""
    key = jax.random.PRNGKey(0)
    add_random_cell = jax.jit(game_2048._add_random_cell)
    updated_board = add_random_cell(board, key)

    # Check that a new value is added in an empty cell.
    assert jax.numpy.count_nonzero(updated_board) == jax.numpy.count_nonzero(board) + 1
    # Check that the new value added is 1 or 2.
    new_value_is_one = jnp.sum(updated_board == 1) == (jnp.sum(board == 1) + 1)
    new_value_is_two = jnp.sum(updated_board == 2) == (jnp.sum(board == 2) + 1)
    assert new_value_is_one ^ new_value_is_two


def test_game_2048__get_action_mask(game_2048: Game2048, board: Board) -> None:
    """Verify that the action mask generated by `get_action_mask` is correct."""
    action_mask_fn = jax.jit(game_2048._get_action_mask)
    action_mask = action_mask_fn(board)
    expected_action_mask = jnp.array([False, True, True, True])
    assert jnp.array_equal(action_mask, expected_action_mask)


def test_game_2048__does_not_smoke(game_2048: Game2048) -> None:
    """Test that we can run an episode without any errors."""
    check_env_does_not_smoke(game_2048)


def test_game_2048__specs_does_not_smoke(game_2048: Game2048) -> None:
    """Test that we access specs without any errors."""
    check_env_specs_does_not_smoke(game_2048)
