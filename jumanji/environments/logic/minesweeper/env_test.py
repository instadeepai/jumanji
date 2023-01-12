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

from typing import List, Optional, Tuple

import chex
import matplotlib.pyplot as plt
import pytest
from jax import jit
from jax import numpy as jnp
from jax import random

from jumanji.environments.logic.minesweeper.constants import (
    REVEALED_EMPTY_SQUARE_REWARD,
    REVEALED_MINE_OR_INVALID_ACTION_REWARD,
)
from jumanji.environments.logic.minesweeper.env import Minesweeper
from jumanji.environments.logic.minesweeper.types import State
from jumanji.testing.env_not_smoke import check_env_does_not_smoke
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import StepType, TimeStep


def play_and_get_episode_stats(
    env: Minesweeper,
    actions: List[chex.Array],
    step_limit: int,
    force_start_state: Optional[State] = None,
) -> Tuple[List[float], List[StepType], int]:
    state, timestep = jit(env.reset)(random.PRNGKey(0))
    if force_start_state:
        state = force_start_state
    episode_length = 0
    step_fn = jit(env.step)
    collected_rewards = []
    collected_step_types = []
    while not timestep.last():
        state, timestep = step_fn(state, actions[episode_length])
        episode_length += 1
        collected_rewards.append(timestep.reward)
        collected_step_types.append(timestep.step_type)
        if episode_length > step_limit:
            # Exit condition to make sure tests don't enter infinite loop, should not
            # be hit
            raise Exception("Entered infinite loop")
    return collected_rewards, collected_step_types, episode_length


@pytest.mark.parametrize(
    "actions, expected_rewards, expected_step_types",
    [
        (
            [[0, 3], [1, 1], [1, 3], [2, 3], [3, 0], [3, 1], [3, 2], [3, 3]],
            [REVEALED_EMPTY_SQUARE_REWARD] * 8,
            [StepType.MID] * 7 + [StepType.LAST],
        ),
        (
            [[0, 3], [0, 2]],
            [REVEALED_EMPTY_SQUARE_REWARD, REVEALED_MINE_OR_INVALID_ACTION_REWARD],
            [StepType.MID, StepType.LAST],
        ),
        (
            [[0, 3], [0, 3]],
            [REVEALED_EMPTY_SQUARE_REWARD, REVEALED_MINE_OR_INVALID_ACTION_REWARD],
            [StepType.MID, StepType.LAST],
        ),
    ],
)
def test_default_reward_and_done_signals(
    minesweeper_env: Minesweeper,
    manual_start_state: State,
    actions: List[List[int]],
    expected_rewards: List[float],
    expected_step_types: List[StepType],
) -> None:
    """Test whether the default reward is working as expected"""
    actions = [jnp.array(action, dtype=jnp.int32) for action in actions]
    rewards, step_types, episode_length = play_and_get_episode_stats(
        env=minesweeper_env,
        actions=actions,
        step_limit=len(actions),
        force_start_state=manual_start_state,
    )
    assert rewards == expected_rewards
    assert step_types == expected_step_types
    assert episode_length == len(actions)


def test_minesweeper_env_reset(minesweeper_env: Minesweeper) -> None:
    """Validates the jitted reset of the environment."""
    reset_fn = jit(minesweeper_env.reset)
    key = random.PRNGKey(0)
    state, timestep = reset_fn(key)
    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)
    assert state.step_count == 0
    assert state.flat_mine_locations.shape == (minesweeper_env.num_mines,)
    assert timestep.observation.num_mines == minesweeper_env.num_mines
    assert state.board.shape == (
        minesweeper_env.board_height,
        minesweeper_env.board_width,
    )
    assert jnp.array_equal(state.board, timestep.observation.board)
    assert timestep.observation.step_count == 0
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # reset function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(state)


def test_minesweeper_env_step(minesweeper_env: Minesweeper) -> None:
    """Validates the jitted step of the environment."""
    chex.clear_trace_counter()
    step_fn = chex.assert_max_traces(minesweeper_env.step, n=2)
    step_fn = jit(step_fn)
    key = random.PRNGKey(0)
    state, timestep = jit(minesweeper_env.reset)(key)
    # For this board, this action will be a non-mined square
    action = minesweeper_env.action_spec().generate_value()
    next_state, next_timestep = step_fn(state, action)

    # Check that the state has changed
    assert not jnp.array_equal(next_state.board, state.board)
    assert next_timestep.observation.num_mines == timestep.observation.num_mines
    assert next_state.step_count == 1
    assert next_timestep.observation.step_count == 1
    assert jnp.array_equal(next_state.board, next_timestep.observation.board)

    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # step function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(next_state)

    next_next_state, next_next_timestep = step_fn(next_state, action)

    # Check that the state has changed, since we took the same action twice
    assert jnp.array_equal(next_next_state.board, next_state.board)
    assert (
        next_next_timestep.observation.num_mines == next_timestep.observation.num_mines
    )
    assert next_next_state.step_count == 2
    assert next_next_timestep.observation.step_count == 2
    assert jnp.array_equal(next_next_state.board, next_next_timestep.observation.board)


def test_minesweeper_env_does_not_smoke(minesweeper_env: Minesweeper) -> None:
    """Test that we can run an episode without any errors."""
    check_env_does_not_smoke(env=minesweeper_env)


def test_minesweeper_env_render(
    monkeypatch: pytest.MonkeyPatch, minesweeper_env: Minesweeper
) -> None:
    """Check that the render method builds the figure but does not display it."""
    monkeypatch.setattr(plt, "show", lambda fig: None)
    state, timestep = jit(minesweeper_env.reset)(random.PRNGKey(0))
    minesweeper_env.render(state)
    minesweeper_env.close()
    action = minesweeper_env.action_spec().generate_value()
    state, timestep = jit(minesweeper_env.step)(state, action)
    minesweeper_env.render(state)
    minesweeper_env.close()


def test_minesweeper_env_done_invalid_action(minesweeper_env: Minesweeper) -> None:
    """Test that the strict done signal is sent correctly"""
    # Note that this action corresponds to not stepping on a mine
    action = minesweeper_env.action_spec().generate_value()
    *_, episode_length = play_and_get_episode_stats(
        env=minesweeper_env, actions=[action for _ in range(10)], step_limit=10
    )
    assert episode_length == 2


def test_minesweeper_env_solved(minesweeper_env: Minesweeper) -> None:
    """Solve the game and verify that things are as expected"""
    state, timestep = jit(minesweeper_env.reset)(random.PRNGKey(0))
    step_fn = jit(minesweeper_env.step)
    collected_rewards = []
    collected_step_types = []
    for i in range(minesweeper_env.board_height):
        for j in range(minesweeper_env.board_width):
            flat_location = i * minesweeper_env.board_width + j
            if flat_location in state.flat_mine_locations:
                continue
            action = jnp.array([i, j], dtype=jnp.int32)
            state, timestep = step_fn(state, action)
            collected_rewards.append(timestep.reward)
            collected_step_types.append(timestep.step_type)
    expected_episode_length = (
        minesweeper_env.board_height * minesweeper_env.board_width
        - minesweeper_env.num_mines
    )
    assert collected_rewards == [REVEALED_EMPTY_SQUARE_REWARD] * expected_episode_length
    assert collected_step_types == [StepType.MID] * (expected_episode_length - 1) + [
        StepType.LAST
    ]
