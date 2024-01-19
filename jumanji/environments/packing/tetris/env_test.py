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
from jax import numpy as jnp

from jumanji.environments.packing.tetris.env import Tetris
from jumanji.environments.packing.tetris.types import State
from jumanji.testing.env_not_smoke import (
    check_env_does_not_smoke,
    check_env_specs_does_not_smoke,
)
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import TimeStep


@pytest.fixture
def tetris_env() -> Tetris:
    return Tetris(num_rows=6, num_cols=6)


@pytest.fixture
def grid() -> chex.Array:
    """Random grid"""
    grid = jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    return grid


def test_tetris_env_reset(tetris_env: Tetris) -> None:
    """Validates the jitted reset of the environment."""
    reset_fn = jax.jit(chex.assert_max_traces(tetris_env.reset, n=1))
    key = jax.random.PRNGKey(0)
    _ = reset_fn(key)
    # Call again to check it does not compile twice.
    state, timestep = reset_fn(key)
    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)
    assert state.grid_padded.sum() == 0
    assert state.grid_padded.shape[0] == (timestep.observation.grid.shape[0] + 3)
    assert state.grid_padded.shape[1] == (timestep.observation.grid.shape[1] + 3)
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # reset function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(state)
    assert_is_jax_array_tree(timestep)


def test_tetris_env_step(tetris_env: Tetris) -> None:
    """Validates the jitted step of the environment."""
    chex.clear_trace_counter()
    step_fn = jax.jit(chex.assert_max_traces(tetris_env.step, n=1))
    key = jax.random.PRNGKey(0)
    state, timestep = tetris_env.reset(key)
    action = (0, 4)
    step_fn(state, action)
    step_fn(state, action)
    step_fn(state, action)
    action = (0, 0)
    next_state, next_timestep = step_fn(state, action)
    # Check that the state has changed
    assert not jnp.array_equal(next_state.grid_padded, state.grid_padded)
    assert next_state.grid_padded.sum() == state.grid_padded.sum() + 4
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # step function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(next_state)
    assert_is_jax_array_tree(next_timestep)


def test_rotate(tetris_env: Tetris) -> None:
    """Test the jited rotate method"""
    rotate_fn = jax.jit(tetris_env._rotate)
    tetromino = rotate_fn(2, 0)
    expected_tetromino = jnp.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ]
    )
    assert tetromino.shape == (4, 4)
    assert (tetromino == expected_tetromino).all()


def test_calculate_action_mask(tetris_env: Tetris, grid: chex.Array) -> None:
    """Test the jited rotate method"""
    action_mask = tetris_env._calculate_action_mask(grid, 0)
    expected_action_mask = jnp.array(
        [
            [True, False, False, True, True, False],
            [True, True, True, False, False, False],
            [True, False, False, True, True, False],
            [True, True, True, False, False, False],
        ]
    )
    assert (action_mask == expected_action_mask).all()


def test_tetris__does_not_smoke(tetris_env: Tetris) -> None:
    """Test that we can run an episode without any errors."""
    check_env_does_not_smoke(tetris_env)


def test_tetris__specs_does_not_smoke(tetris_env: Tetris) -> None:
    """Test that we can access specs without any errors."""
    check_env_specs_does_not_smoke(tetris_env)
