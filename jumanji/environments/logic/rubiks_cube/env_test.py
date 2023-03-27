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
import matplotlib.animation
import matplotlib.pyplot as plt
import pytest
import pytest_mock

from jumanji.environments.logic.rubiks_cube.env import RubiksCube
from jumanji.environments.logic.rubiks_cube.types import State
from jumanji.testing.env_not_smoke import check_env_does_not_smoke
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import TimeStep


def test_rubiks_cube__reset(rubiks_cube: RubiksCube) -> None:
    """Validates the jitted reset of the environment."""
    chex.clear_trace_counter()
    reset_fn = jax.jit(chex.assert_max_traces(rubiks_cube.reset, n=1))
    key = jax.random.PRNGKey(0)
    _ = reset_fn(key)
    state, timestep = reset_fn(key)
    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)
    assert state.step_count == 0
    assert jnp.array_equal(state.cube, timestep.observation.cube)
    assert timestep.observation.step_count == 0
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # reset function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(state)


def test_rubiks_cube__step(rubiks_cube: RubiksCube) -> None:
    """Validates the jitted step of the environment."""
    chex.clear_trace_counter()
    step_fn = jax.jit(chex.assert_max_traces(rubiks_cube.step, n=1))
    key = jax.random.PRNGKey(0)
    state, timestep = rubiks_cube.reset(key)
    action = rubiks_cube.action_spec().generate_value()
    next_state, next_timestep = step_fn(state, action)

    # Check that the state has changed
    assert not jnp.array_equal(next_state.cube, state.cube)
    assert next_state.step_count == 1
    assert next_timestep.observation.step_count == 1
    assert jnp.array_equal(next_state.cube, next_timestep.observation.cube)

    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # step function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(next_state)

    next_next_state, next_next_timestep = step_fn(next_state, action)

    # Check that the state has changed
    assert not jnp.array_equal(next_next_state.cube, state.cube)
    assert not jnp.array_equal(next_next_state.cube, next_state.cube)
    assert next_next_state.step_count == 2
    assert next_next_timestep.observation.step_count == 2
    assert jnp.array_equal(next_next_state.cube, next_next_timestep.observation.cube)


@pytest.mark.parametrize("cube_size", [3, 4, 5])
def test_rubiks_cube__does_not_smoke(cube_size: int) -> None:
    """Test that we can run an episode without any errors."""
    env = RubiksCube(cube_size=cube_size, time_limit=10, num_scrambles_on_reset=5)
    check_env_does_not_smoke(env)


def test_rubiks_cube__render(
    monkeypatch: pytest.MonkeyPatch, rubiks_cube: RubiksCube
) -> None:
    """Test that the render method builds the figure (but does not display it)."""
    monkeypatch.setattr(plt, "show", lambda fig: None)
    state, timestep = rubiks_cube.reset(jax.random.PRNGKey(0))
    rubiks_cube.render(state)
    rubiks_cube.close()
    action = rubiks_cube.action_spec().generate_value()
    state, timestep = rubiks_cube.step(state, action)
    rubiks_cube.render(state)
    rubiks_cube.close()


@pytest.mark.parametrize("time_limit", [3, 4, 5])
def test_rubiks_cube__done(time_limit: int) -> None:
    """Test that the done signal is sent correctly."""
    env = RubiksCube(time_limit=time_limit)
    state, timestep = env.reset(jax.random.PRNGKey(0))
    action = env.action_spec().generate_value()
    episode_length = 0
    step_fn = jax.jit(env.step)
    while not timestep.last():
        state, timestep = step_fn(state, action)
        episode_length += 1
        if episode_length > 10:
            # Exit condition to make sure tests don't enter infinite loop, should not be hit
            raise Exception("Entered infinite loop")
    assert episode_length == time_limit


def test_rubiks_cube__animate(
    rubiks_cube: RubiksCube, mocker: pytest_mock.MockerFixture
) -> None:
    """Test that the `animate` method creates the animation correctly (but does not display it)."""
    states = mocker.MagicMock()
    animation = rubiks_cube.animate(states)
    assert isinstance(animation, matplotlib.animation.Animation)
