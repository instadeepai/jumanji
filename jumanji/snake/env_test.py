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
from jax import random

from jumanji.snake import Snake, State
from jumanji.snake.types import Position
from jumanji.testing.env_not_smoke import check_env_does_not_smoke
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import TimeStep


@pytest.fixture(scope="module")
def snake_env() -> Snake:
    """Instantiates a default Snake environment."""
    return Snake(6, 6)


@pytest.mark.parametrize("snake_env", [()], indirect=True)
def test_snake__reset(snake_env: Snake) -> None:
    """Validates the jitted reset of the environment."""
    reset_fn = jax.jit(snake_env.reset)
    key1, key2 = random.PRNGKey(0), random.PRNGKey(1)
    state1, timestep1, _ = reset_fn(key1)
    state2, timestep2, _ = reset_fn(key2)
    assert isinstance(timestep1, TimeStep)
    assert isinstance(state1, State)
    assert state1.step == 0
    assert state1.length == 1
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # reset function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(state1)
    # Check random initialization
    assert state1.head_pos != state2.head_pos
    assert state1.fruit_pos != state2.fruit_pos
    assert not jnp.all(state1.key == state2.key)
    assert not jnp.all(state1.body_state == state2.body_state)


@pytest.mark.parametrize("snake_env", [()], indirect=True)
def test_snake__step(snake_env: Snake) -> None:
    """Validates the jitted step function of the environment."""
    step_fn = jax.jit(snake_env.step)
    state_key, action_key = random.split(random.PRNGKey(10))
    state, timestep, _ = snake_env.reset(state_key)
    # Sample two different actions
    action1, action2 = random.choice(
        action_key,
        jnp.arange(snake_env.action_spec()._num_values),
        shape=(2,),
        replace=False,
    )
    new_state1, timestep1, _ = step_fn(state, action1)
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # step function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(new_state1)
    # Check that the state has changed
    assert new_state1.step != state.step
    assert new_state1.head_pos != state.head_pos
    # Check that two different actions lead to two different states
    new_state2, timestep2, _ = step_fn(state, action2)
    assert new_state1.head_pos != new_state2.head_pos
    # Check that the state update and timestep creation work as expected
    row, col = tuple(state.head_pos)
    body = timestep.observation[..., 0].at[(row, col)].set(False)
    moves = {
        0: (Position(row - 1, col), body.at[(row - 1, col)].set(True)),  # Up
        1: (Position(row, col + 1), body.at[(row, col + 1)].set(True)),  # Right
        2: (Position(row + 1, col), body.at[(row + 1, col)].set(True)),  # Down
        3: (Position(row, col - 1), body.at[(row, col - 1)].set(True)),  # Left
    }
    for action, (new_position, new_body) in moves.items():
        new_state, timestep, _ = step_fn(state, action)
        assert new_state.head_pos == new_position
        assert jnp.all(timestep.observation[..., 0] == new_body)


@pytest.mark.parametrize("snake_env", [()], indirect=True)
def test_snake__does_not_smoke(snake_env: Snake) -> None:
    """Test that we can run an episode without any errors."""
    check_env_does_not_smoke(snake_env)


def test_update_head_pos() -> None:
    """Validates update_head_pos method from utils.
    Checks that starting from a certain position, taking some actions
    lead to the correct new positions.
    """
    head_pos = Position(3, 5)
    actions = jnp.array([0, 1, 2, 3], int)
    updated_head_poss = [Snake.update_head_pos(head_pos, action) for action in actions]
    next_head_poss = [
        Position(2, 5),
        Position(3, 6),
        Position(4, 5),
        Position(3, 4),
    ]
    assert next_head_poss == updated_head_poss


@pytest.mark.parametrize("snake_env", [()], indirect=True)
def test_snake__no_nan(snake_env: Snake) -> None:
    """Validates that no nan is encountered in either the state or the observation throughout an
    episode. Checks both exiting from the top and right of the board as jax out-of-bounds indices
    have different behaviors if positive or negative.
    """
    reset_fn = jax.jit(snake_env.reset)
    step_fn = jax.jit(snake_env.step)
    key = random.PRNGKey(0)
    # Check exiting the board to the top
    state, timestep, _ = reset_fn(key)
    chex.assert_tree_all_finite((state, timestep))
    while not timestep.last():
        state, timestep, _ = step_fn(state, action=0)
        chex.assert_tree_all_finite((state, timestep))
    # Check exiting the board to the right
    state, timestep, _ = reset_fn(key)
    chex.assert_tree_all_finite((state, timestep))
    while not timestep.last():
        state, timestep, _ = step_fn(state, action=1)
        chex.assert_tree_all_finite((state, timestep))


def test_snake__render(monkeypatch: pytest.MonkeyPatch, snake_env: Snake) -> None:
    """Check that the render method builds the figure but does not display it."""
    monkeypatch.setattr(plt, "show", lambda fig: None)
    state, timestep, _ = snake_env.reset(jax.random.PRNGKey(0))
    while not timestep.last():
        state, timestep, _ = snake_env.step(
            state, action=snake_env.action_spec().generate_value()
        )
        snake_env.render(state)
    snake_env.close()


def test_snake__animation(snake_env: Snake, tmpdir: py.path.local) -> None:
    """Check that the animation method creates the animation correctly and can save to a gif."""
    state, timestep, _ = snake_env.reset(jax.random.PRNGKey(0))
    states = [state]
    while not timestep.last():
        state, timestep, _ = snake_env.step(
            state, action=snake_env.action_spec().generate_value()
        )
        states.append(state)
    animation = snake_env.animation(states)
    assert isinstance(animation, matplotlib.animation.Animation)

    path = str(tmpdir.join("/anim.gif"))
    animation.save(path, writer=matplotlib.animation.PillowWriter(fps=10), dpi=60)
