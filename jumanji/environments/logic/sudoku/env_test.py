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
import matplotlib.animation
import matplotlib.pyplot as plt
import pytest
import pytest_mock
from jax import numpy as jnp

from jumanji.environments.logic.sudoku.env import Sudoku
from jumanji.environments.logic.sudoku.types import State
from jumanji.testing.env_not_smoke import check_env_does_not_smoke
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import TimeStep


def test_sudoku__reset(sudoku_env: Sudoku) -> None:
    """Validates the jitted reset of the environment."""
    reset_fn = chex.assert_max_traces(sudoku_env.reset, n=1)
    reset_fn = jax.jit(reset_fn)

    key = jax.random.PRNGKey(0)
    state, timestep = reset_fn(key)

    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)

    assert jnp.array_equal(state.board, timestep.observation.board)
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # reset function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(state)


def test_sudoku__step(sudoku_env: Sudoku) -> None:
    """Validates the jitted step of the environment."""
    chex.clear_trace_counter()
    step_fn = chex.assert_max_traces(sudoku_env.step, n=1)
    step_fn = jax.jit(step_fn)
    key = jax.random.PRNGKey(0)
    state, timestep = jax.jit(sudoku_env.reset)(key)

    action = sudoku_env.action_spec().generate_value()
    next_state, next_timestep = step_fn(state, action)

    # Check that the state has changed
    assert not jnp.array_equal(next_state.board, state.board)
    assert jnp.array_equal(next_state.board, next_timestep.observation.board)

    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # step function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(next_state)

    next_next_state, next_next_timestep = step_fn(next_state, action)

    # Check that the state has changed, since we took the same action twice
    assert jnp.array_equal(next_next_state.board, next_next_timestep.observation.board)


def test_sudoku__does_not_smoke(sudoku_env: Sudoku) -> None:
    """Test that we can run an episode without any errors."""
    check_env_does_not_smoke(env=sudoku_env)


def test_sudoku__render(monkeypatch: pytest.MonkeyPatch, sudoku_env: Sudoku) -> None:
    """Check that the render method builds the figure but does not display it."""
    monkeypatch.setattr(plt, "show", lambda fig: None)
    state, timestep = jax.jit(sudoku_env.reset)(jax.random.PRNGKey(0))
    sudoku_env.render(state)
    sudoku_env.close()
    action = sudoku_env.action_spec().generate_value()
    state, timestep = jax.jit(sudoku_env.step)(state, action)
    sudoku_env.render(state)
    sudoku_env.close()


def test_sudoku_animation(
    sudoku_env: Sudoku, mocker: pytest_mock.MockerFixture
) -> None:
    """Check that the animation method creates the animation correctly."""
    states = mocker.MagicMock()
    animation = sudoku_env.animate(states)
    assert isinstance(animation, matplotlib.animation.Animation)
