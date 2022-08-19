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
import pytest
from chex import Array
from jax import numpy as jnp
from jax import random

from jumanji.connect4.constants import BOARD_HEIGHT, BOARD_WIDTH
from jumanji.connect4.env import Connect4, compute_reward
from jumanji.connect4.types import State
from jumanji.testing.env_not_smoke import check_env_does_not_smoke
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import TimeStep


@pytest.fixture
def connect4_env() -> Connect4:
    """Instantiates a default Connect4 environment."""
    return Connect4()


def test_connect4__reset(connect4_env: Connect4, empty_board: Array) -> None:
    """Validates the jitted reset of the environment."""
    reset_fn = jax.jit(connect4_env.reset)
    key = random.PRNGKey(0)
    state, timestep = reset_fn(key)
    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)
    assert state.current_player == 0
    assert timestep.extras["current_player"] == 0  # type: ignore
    assert jnp.array_equal(state.board, empty_board)
    assert jnp.array_equal(
        timestep.observation.action_mask, jnp.ones((BOARD_WIDTH,), dtype=jnp.int8)
    )
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # reset function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(state)


def test_connect4__step(connect4_env: Connect4, empty_board: Array) -> None:
    """Validates the jitted step of the environment."""
    chex.clear_trace_counter()

    step_fn = chex.assert_max_traces(connect4_env.step, n=1)
    step_fn = jax.jit(step_fn)

    key = random.PRNGKey(0)
    state, timestep = connect4_env.reset(key)

    action = jnp.array(0)

    new_state, next_timestep = step_fn(state, action)

    # Check that the state has changed
    assert not jnp.array_equal(new_state.board, state.board)
    assert new_state.current_player != state.current_player

    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # step function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(new_state)

    # Check token was inserted as expected
    assert new_state.board[BOARD_HEIGHT - 1, action] == -1

    # New step
    state = new_state

    new_state, next_timestep = step_fn(state, action)

    # Check that the state has changed
    assert not jnp.array_equal(new_state.board, state.board)
    assert new_state.current_player != state.current_player

    # Check token was inserted as expected
    assert new_state.board[BOARD_HEIGHT - 1, action] == 1
    assert new_state.board[BOARD_HEIGHT - 2, action] == -1


def test_connect4__does_not_smoke(connect4_env: Connect4) -> None:
    """Test that we can run an episode without any errors."""
    check_env_does_not_smoke(connect4_env)


@pytest.mark.parametrize(
    "winning, invalid, expected_reward",
    [(True, True, -1), (True, False, 1), (False, False, 0), (False, True, -1)],
)
def test_connect4__compute_reward(
    winning: bool, invalid: bool, expected_reward: int
) -> None:
    """Check each combination of the compute_reward function"""
    winning = jnp.array(winning)
    invalid = jnp.array(invalid)
    reward = compute_reward(invalid, winning)
    assert reward == expected_reward


def test_connect4__invalid_action(connect4_env: Connect4) -> None:
    """Checks that an invalid action leads to a termination
    and the appropriate reward is received
    """
    key = random.PRNGKey(0)
    state, timestep = connect4_env.reset(key)
    action = jnp.array(0)

    # legal actions
    for _ in range(BOARD_HEIGHT):
        state, timestep = connect4_env.step(state, action)

    # check that the action is flagged as illegal
    assert timestep.observation.action_mask[0] == 0
    # check the other actions are still legal
    assert jnp.all(timestep.observation.action_mask[1:] == 1)

    bad_player = state.current_player
    good_player = (bad_player + 1) % 2

    # do the illegal action
    state, timestep = connect4_env.step(state, action)

    assert timestep.last()
    assert timestep.reward[bad_player] == -1
    assert timestep.reward[good_player] == 1


def test_connect4__draw(connect4_env: Connect4) -> None:
    """Checks that an invalid action leads to a termination
    and the appropriate reward is received
    """
    key = random.PRNGKey(0)
    state, timestep = connect4_env.reset(key)

    # create a full board

    # this is a hack to ensure the board passed is not winning deterministically
    # and yet flagged as full.
    full_row = jnp.tile(jnp.array([-1, 1], dtype=jnp.int8), BOARD_WIDTH)[:BOARD_WIDTH]
    full_column = jnp.tile(jnp.array([-1, 1], dtype=jnp.int8), BOARD_HEIGHT)[
        :BOARD_HEIGHT
    ]

    full_board = state.board.at[0, :].set(full_row)
    full_board = full_board.at[:, 0].set(full_column)

    # free one place
    full_board = full_board.at[0, 0].set(0)
    state.board = full_board

    # do the last action
    action = jnp.array(0)
    state, timestep = connect4_env.step(state, action)

    # check for termination
    assert timestep.last()

    # check no rewards
    assert jnp.all(timestep.reward == 0)


def test_connect4__render(connect4_env: Connect4, empty_board: Array) -> None:
    """Checks that render functions correctly and returns the expected string"""
    key = random.PRNGKey(0)
    state, timestep = connect4_env.reset(key)

    render = connect4_env.render(state)

    assert "Current player: 0" in render

    expected_board_render = str(empty_board)

    assert expected_board_render in render
