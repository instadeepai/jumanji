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
import matplotlib.pyplot as plt
import pytest
from jax import jit
from jax import numpy as jnp
from jax import random, vmap

from jumanji.environments.logic.rubiks_cube.constants import CubeMovementAmount, Face
from jumanji.environments.logic.rubiks_cube.env import RubiksCube
from jumanji.environments.logic.rubiks_cube.types import State
from jumanji.testing.env_not_smoke import check_env_does_not_smoke
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import TimeStep


@pytest.mark.parametrize("cube_size", [2, 3, 4, 5])
def test_flatten(cube_size: int) -> None:
    """Test that flattening and unflattening actions are inverse to each other"""
    env = RubiksCube(cube_size=cube_size)
    flat_actions = jnp.arange(
        len(Face) * (cube_size // 2) * len(CubeMovementAmount), dtype=jnp.int16
    )
    faces = jnp.arange(len(Face), dtype=jnp.int16)
    depths = jnp.arange(cube_size // 2, dtype=jnp.int16)
    amounts = jnp.arange(len(CubeMovementAmount), dtype=jnp.int16)
    unflat_actions = jnp.stack(
        [
            jnp.repeat(faces, len(CubeMovementAmount) * (cube_size // 2)),
            jnp.concatenate(
                [jnp.repeat(depths, len(CubeMovementAmount)) for _ in Face]
            ),
            jnp.concatenate([amounts for _ in range(len(Face) * (cube_size // 2))]),
        ]
    )
    assert jnp.array_equal(unflat_actions, env._unflatten_action(flat_actions))
    assert jnp.array_equal(flat_actions, env._flatten_action(unflat_actions))


def test_scramble_on_reset(
    rubiks_cube_env: RubiksCube, expected_scramble_result: chex.Array
) -> None:
    """Test that the environment reset is performing correctly when given a particular scramble
    (chosen manually)"""
    amount_to_index = {
        CubeMovementAmount.CLOCKWISE: 0,
        CubeMovementAmount.ANTI_CLOCKWISE: 1,
        CubeMovementAmount.HALF_TURN: 2,
    }
    unflattened_sequence = jnp.array(
        [
            [Face.UP.value, 0, amount_to_index[CubeMovementAmount.CLOCKWISE]],
            [Face.LEFT.value, 0, amount_to_index[CubeMovementAmount.HALF_TURN]],
            [Face.DOWN.value, 0, amount_to_index[CubeMovementAmount.ANTI_CLOCKWISE]],
            [Face.UP.value, 0, amount_to_index[CubeMovementAmount.HALF_TURN]],
            [Face.BACK.value, 0, amount_to_index[CubeMovementAmount.ANTI_CLOCKWISE]],
            [Face.RIGHT.value, 0, amount_to_index[CubeMovementAmount.CLOCKWISE]],
            [Face.FRONT.value, 0, amount_to_index[CubeMovementAmount.CLOCKWISE]],
            [Face.RIGHT.value, 0, amount_to_index[CubeMovementAmount.ANTI_CLOCKWISE]],
            [Face.LEFT.value, 0, amount_to_index[CubeMovementAmount.ANTI_CLOCKWISE]],
            [Face.BACK.value, 0, amount_to_index[CubeMovementAmount.HALF_TURN]],
            [Face.FRONT.value, 0, amount_to_index[CubeMovementAmount.ANTI_CLOCKWISE]],
            [Face.UP.value, 0, amount_to_index[CubeMovementAmount.CLOCKWISE]],
            [Face.DOWN.value, 0, amount_to_index[CubeMovementAmount.CLOCKWISE]],
        ],
        dtype=jnp.int16,
    )
    flat_sequence = jnp.array(
        [0, 14, 16, 2, 10, 6, 3, 7, 13, 11, 4, 0, 15], dtype=jnp.int16
    )
    assert jnp.array_equal(
        unflattened_sequence.transpose(),
        rubiks_cube_env._unflatten_action(action=flat_sequence),
    )
    assert jnp.array_equal(
        flat_sequence, vmap(rubiks_cube_env._flatten_action)(unflattened_sequence)
    )
    cube = rubiks_cube_env._scramble_solved_cube(flat_actions_in_scramble=flat_sequence)
    assert jnp.array_equal(expected_scramble_result, cube)


def test_rubiks_cube_env_reset(rubiks_cube_env: RubiksCube) -> None:
    """Validates the jitted reset of the environment."""
    reset_fn = jit(rubiks_cube_env.reset)
    key = random.PRNGKey(0)
    state, timestep = reset_fn(key)
    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)
    assert state.step_count == 0
    assert state.action_history.shape == (
        rubiks_cube_env.num_scrambles_on_reset + rubiks_cube_env.step_limit,
        3,
    )
    action_history_index = rubiks_cube_env.num_scrambles_on_reset
    assert jnp.all(jnp.equal(state.action_history[action_history_index:], 0))
    assert state.action_history.min() >= 0
    assert state.action_history[:, 0].max() < len(Face)
    assert state.action_history[:, 1].max() < rubiks_cube_env.cube_size // 2
    assert state.action_history[:, 2].max() < len(CubeMovementAmount)
    assert jnp.array_equal(state.cube, timestep.observation.cube)
    assert timestep.observation.step_count == 0
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # reset function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(state)


def test_rubiks_cube_env_step(rubiks_cube_env: RubiksCube) -> None:
    """Validates the jitted step of the environment."""
    chex.clear_trace_counter()
    step_fn = chex.assert_max_traces(rubiks_cube_env.step, n=2)
    step_fn = jit(step_fn)
    key = random.PRNGKey(0)
    state, timestep = jit(rubiks_cube_env.reset)(key)
    action = rubiks_cube_env.action_spec().generate_value()
    next_state, next_timestep = step_fn(state, action)

    # Check that the state has changed
    assert not jnp.array_equal(next_state.cube, state.cube)
    assert next_state.step_count == 1
    assert next_timestep.observation.step_count == 1
    assert jnp.array_equal(next_state.cube, next_timestep.observation.cube)
    assert next_state.action_history.shape == (
        rubiks_cube_env.num_scrambles_on_reset + rubiks_cube_env.step_limit,
        3,
    )
    action_history_index = rubiks_cube_env.num_scrambles_on_reset + 1
    assert jnp.all(jnp.equal(next_state.action_history[action_history_index:], 0))

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
    assert next_next_state.action_history.shape == (
        rubiks_cube_env.num_scrambles_on_reset + rubiks_cube_env.step_limit,
        3,
    )
    action_history_index = rubiks_cube_env.num_scrambles_on_reset + 2
    assert jnp.all(jnp.equal(next_next_state.action_history[action_history_index:], 0))


@pytest.mark.parametrize("cube_size", [3, 4, 5])
def test_rubiks_cube_env_does_not_smoke(cube_size: int) -> None:
    """Test that we can run an episode without any errors."""
    check_env_does_not_smoke(env=RubiksCube(cube_size=cube_size, step_limit=10))


def test_rubiks_cube_env_render(
    monkeypatch: pytest.MonkeyPatch, rubiks_cube_env: RubiksCube
) -> None:
    """Check that the render method builds the figure but does not display it."""
    monkeypatch.setattr(plt, "show", lambda fig: None)
    state, timestep = jit(rubiks_cube_env.reset)(random.PRNGKey(0))
    rubiks_cube_env.render(state)
    rubiks_cube_env.close()
    action = rubiks_cube_env.action_spec().generate_value()
    state, timestep = jit(rubiks_cube_env.step)(state, action)
    rubiks_cube_env.render(state)
    rubiks_cube_env.close()


@pytest.mark.parametrize("step_limit", [3, 4, 5])
def test_rubiks_cube_env_done(step_limit: int) -> None:
    """Test that the done signal is sent correctly"""
    env = RubiksCube(step_limit=step_limit)
    state, timestep = jit(env.reset)(random.PRNGKey(0))
    action = env.action_spec().generate_value()
    episode_length = 0
    step_fn = jit(env.step)
    while not timestep.last():
        state, timestep = step_fn(state, action)
        episode_length += 1
        if episode_length > 10:
            # Exit condition to make sure tests don't enter infinite loop, should not be hit
            raise Exception("Entered infinite loop")
    assert episode_length == step_limit
