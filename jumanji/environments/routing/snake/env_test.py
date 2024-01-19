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
import py
import pytest

from jumanji.environments.routing.snake.env import Snake, State
from jumanji.environments.routing.snake.types import Position
from jumanji.testing.env_not_smoke import (
    check_env_does_not_smoke,
    check_env_specs_does_not_smoke,
)
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import TimeStep


@pytest.fixture(scope="module")
def snake() -> Snake:
    """Instantiates a default Snake environment."""
    return Snake(6, 6)


def test_snake__reset(snake: Snake) -> None:
    """Validates the jitted reset of the environment."""
    reset_fn = jax.jit(chex.assert_max_traces(snake.reset, n=1))
    state1, timestep1 = reset_fn(jax.random.PRNGKey(1))
    state2, timestep2 = reset_fn(jax.random.PRNGKey(2))
    assert isinstance(timestep1, TimeStep)
    assert isinstance(state1, State)
    assert state1.step_count == 0
    assert state1.length == 1
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # reset function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(state1)
    # Check random initialization
    assert state1.head_position != state2.head_position
    assert state1.fruit_position != state2.fruit_position
    assert not jnp.array_equal(state1.key, state2.key)
    assert not jnp.array_equal(state1.key, state2.key)


def test_snake__step(snake: Snake) -> None:
    """Validates the jitted step function of the environment."""
    step_fn = jax.jit(chex.assert_max_traces(snake.step, n=1))
    state_key, action_key = jax.random.split(jax.random.PRNGKey(0))
    state, timestep = snake.reset(state_key)
    # Sample two different actions
    action1, action2 = jax.random.choice(
        action_key,
        jnp.arange(snake.action_spec()._num_values),
        shape=(2,),
        replace=False,
    )
    state1, timestep1 = step_fn(state, action1)
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # step function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(state1)
    # Check that the state has changed
    assert state1.step_count != state.step_count
    assert state1.head_position != state.head_position
    # Check that two different actions lead to two different states
    state2, timestep2 = step_fn(state, action2)
    assert state1.head_position != state2.head_position
    # Check that the state update and timestep creation work as expected
    row, col = tuple(state.head_position)
    body = timestep.observation.grid[..., 0].at[(row, col)].set(False)
    moves = {
        0: (Position(row - 1, col), body.at[(row - 1, col)].set(True)),  # Up
        1: (Position(row, col + 1), body.at[(row, col + 1)].set(True)),  # Right
        2: (Position(row + 1, col), body.at[(row + 1, col)].set(True)),  # Down
        3: (Position(row, col - 1), body.at[(row, col - 1)].set(True)),  # Left
    }
    for action, (new_position, new_body) in moves.items():
        new_state, timestep = step_fn(state, jnp.asarray(action, jnp.int32))
        assert new_state.head_position == new_position
        assert jnp.all(timestep.observation.grid[..., 0] == new_body)


def test_snake__does_not_smoke(snake: Snake) -> None:
    """Test that we can run an episode without any errors."""
    check_env_does_not_smoke(snake)


def test_snake__specs_does_not_smoke(snake: Snake) -> None:
    """Test that we can access specs without any errors."""
    check_env_specs_does_not_smoke(snake)


def test_update_head_position(snake: Snake) -> None:
    """Validates _update_head_position method.
    Checks that starting from a certain position, taking some actions
    lead to the correct new positions.
    """
    head_position = Position(3, 5)
    actions = jnp.array([0, 1, 2, 3], int)
    updated_heads_positions = [
        snake._update_head_position(head_position, action) for action in actions
    ]
    next_heads_positions = [
        Position(2, 5),
        Position(3, 6),
        Position(4, 5),
        Position(3, 4),
    ]
    assert next_heads_positions == updated_heads_positions


def test_snake__no_nan(snake: Snake) -> None:
    """Validates that no nan is encountered in either the state or the observation throughout an
    episode. Checks both exiting from the top and right of the board as jax out-of-bounds indices
    have different behaviors if positive or negative.
    """
    reset_fn = jax.jit(snake.reset)
    step_fn = jax.jit(snake.step)
    key = jax.random.PRNGKey(0)
    # Check exiting the board to the top
    state, timestep = reset_fn(key)
    chex.assert_tree_all_finite((state, timestep))
    while not timestep.last():
        state, timestep = step_fn(state, action=0)
        chex.assert_tree_all_finite((state, timestep))
    # Check exiting the board to the right
    state, timestep = reset_fn(key)
    chex.assert_tree_all_finite((state, timestep))
    while not timestep.last():
        state, timestep = step_fn(state, action=1)
        chex.assert_tree_all_finite((state, timestep))


def test_snake__render(monkeypatch: pytest.MonkeyPatch, snake: Snake) -> None:
    """Check that the render method builds the figure but does not display it."""
    monkeypatch.setattr(plt, "show", lambda fig: None)
    step_fn = jax.jit(snake.step)
    state, timestep = snake.reset(jax.random.PRNGKey(0))
    action = snake.action_spec().generate_value()
    state, timestep = step_fn(state, action)
    snake.render(state)
    snake.close()


def test_snake__animation(snake: Snake, tmpdir: py.path.local) -> None:
    """Check that the animation method creates the animation correctly and can save to a gif."""
    step_fn = jax.jit(snake.step)
    state, _ = snake.reset(jax.random.PRNGKey(0))
    states = [state]
    action = snake.action_spec().generate_value()
    state, _ = step_fn(state, action)
    states.append(state)
    animation = snake.animate(states)
    assert isinstance(animation, matplotlib.animation.Animation)

    path = str(tmpdir.join("/anim.gif"))
    animation.save(path, writer=matplotlib.animation.PillowWriter(fps=10), dpi=60)
