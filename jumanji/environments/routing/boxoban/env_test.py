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

import random

import chex
import jax
import jax.numpy as jnp
import pytest

from jumanji.environments.routing.boxoban.constants import AGENT, BOX, TARGET, WALL
from jumanji.environments.routing.boxoban.env import Boxoban
from jumanji.environments.routing.boxoban.generator import (
    DeepMindGenerator,
    SimpleSolveGenerator,
)
from jumanji.environments.routing.boxoban.types import State
from jumanji.testing.env_not_smoke import check_env_does_not_smoke
from jumanji.types import TimeStep


@pytest.fixture(scope="session")
def boxoban() -> Boxoban:
    env = Boxoban(
        generator=DeepMindGenerator(
            difficulty="unfiltered",
            split="train",
            proportion_of_files=0.005,
        )
    )
    return env


@pytest.fixture(scope="session")
def boxoban_simple() -> Boxoban:
    env = Boxoban(generator=SimpleSolveGenerator())
    return env


def test_boxoban__reset(boxoban: Boxoban) -> None:
    chex.clear_trace_counter()
    reset_fn = jax.jit(chex.assert_max_traces(boxoban.reset, n=1))
    key = jax.random.PRNGKey(0)
    state, timestep = reset_fn(key)
    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)
    assert state.step_count == 0
    assert timestep.observation.step_count == 0
    key2 = jax.random.PRNGKey(1)
    state2, timestep2 = reset_fn(key2)
    assert not jnp.array_equal(state2.fixed_grid, state.fixed_grid)
    assert not jnp.array_equal(state2.variable_grid, state.variable_grid)


def test_boxoban__multi_step(boxoban: Boxoban) -> None:
    """Validates the jitted step of the boxoban environment."""
    chex.clear_trace_counter()
    step_fn = jax.jit(chex.assert_max_traces(boxoban.step, n=1))

    # Repeat test for 5 different state initializations
    for j in range(5):
        step_count = 0
        key = jax.random.PRNGKey(j)
        reset_key, step_key = jax.random.split(key)
        state, timestep = boxoban.reset(reset_key)

        # Repeating random step 120 times
        for _ in range(120):
            action = jnp.array(random.randint(0, 4), jnp.int32)
            state, timestep = step_fn(state, action)

            # Check step_count increases after each step
            step_count += 1
            assert state.step_count == step_count
            assert timestep.observation.step_count == step_count

            # Check that the fixed part of the state has not changed
            assert jnp.array_equal(state.fixed_grid, state.fixed_grid)

            # Check that there are always four boxes in the variable grid and 0 elsewhere
            num_boxes = jnp.sum(state.variable_grid == BOX)
            assert num_boxes == jnp.array(4, jnp.int32)

            num_boxes = jnp.sum(state.fixed_grid == BOX)
            assert num_boxes == jnp.array(0, jnp.int32)

            # Check that there are always 4 targets in the fixed grid and 0 elsewhere
            num_targets = jnp.sum(state.variable_grid == TARGET)
            assert num_targets == jnp.array(0, jnp.int32)

            num_targets = jnp.sum(state.fixed_grid == TARGET)
            assert num_targets == jnp.array(4, jnp.int32)

            # Check that there is one agent in variable grid and 0 elsewhere
            num_agents = jnp.sum(state.variable_grid == AGENT)
            assert num_agents == jnp.array(1, jnp.int32)

            num_agents = jnp.sum(state.fixed_grid == AGENT)
            assert num_agents == jnp.array(0, jnp.int32)

            # Check that the grid size remains constant
            assert state.fixed_grid.shape == (10, 10)

            # Check the agent is never in the same location as a wall
            mask_agent = state.variable_grid == AGENT
            mask_wall = state.fixed_grid == WALL
            num_agents_on_wall = jnp.sum(mask_agent & mask_wall)
            assert num_agents_on_wall == jnp.array(0, jnp.int32)

            # Check the boxes are never on a wall
            mask_boxes = state.variable_grid == BOX
            mask_wall = state.fixed_grid == WALL
            num_agents_on_wall = jnp.sum(mask_boxes & mask_wall)
            assert num_agents_on_wall == jnp.array(0, jnp.int32)


def test_boxoban__termination_timelimit(boxoban: Boxoban) -> None:
    """Check that with random actions the environment terminates after
    120 steps"""

    chex.clear_trace_counter()
    step_fn = jax.jit(chex.assert_max_traces(boxoban.step, n=1))

    key = jax.random.PRNGKey(0)
    reset_key, step_key = jax.random.split(key)
    state, timestep = boxoban.reset(reset_key)

    for _ in range(119):
        action = jnp.array(random.randint(0, 4), jnp.int32)
        state, timestep = step_fn(state, action)

        assert not timestep.last()

    action = jnp.array(random.randint(0, 4), jnp.int32)
    state, timestep = step_fn(state, action)

    assert timestep.last()


def test_boxoban__termination_solved(boxoban_simple: Boxoban) -> None:
    """Check that with correct sequence of actions to solve trivial problem,
    the environment terminates"""

    correct_actions = [0, 1, 3] * 3 + [0]
    wrong_actions = [0, 1, 3] * 3 + [1]

    chex.clear_trace_counter()
    step_fn = jax.jit(chex.assert_max_traces(boxoban_simple.step, n=1))

    # Check that environment does terminate with right series of actions
    key = jax.random.PRNGKey(0)
    reset_key, step_key = jax.random.split(key)
    state, timestep = boxoban_simple.reset(reset_key)

    for action in correct_actions:
        assert not timestep.last()

        action = jnp.array(action, jnp.int32)
        state, timestep = step_fn(state, action)

    assert timestep.last()

    # Check that environment does not terminate with wrong series of actions
    key = jax.random.PRNGKey(0)
    reset_key, step_key = jax.random.split(key)
    state, timestep = boxoban_simple.reset(reset_key)

    for action in wrong_actions:
        assert not timestep.last()

        action = jnp.array(action, jnp.int32)
        state, timestep = step_fn(state, action)

    assert not timestep.last()


def test_boxoban__reward_function_solved(boxoban_simple: Boxoban) -> None:
    """Check the reward function is correct when solving the trivial problem.
    Every step should give -0.1, each box added to a target adds 1 and
    solving adds an additional 10"""

    # Correct actions that lead to placing a box every 3 actions
    correct_actions = [0, 1, 3] * 3 + [0]

    chex.clear_trace_counter()
    step_fn = jax.jit(chex.assert_max_traces(boxoban_simple.step, n=1))

    key = jax.random.PRNGKey(0)
    reset_key, step_key = jax.random.split(key)
    state, timestep = boxoban_simple.reset(reset_key)

    for i, action in enumerate(correct_actions):
        action = jnp.array(action, jnp.int32)
        state, timestep = step_fn(state, action)

        if i % 3 == 0 and i != 9:
            assert timestep.reward == jnp.array(0.9, jnp.float32)
        elif i != 9:
            assert timestep.reward == jnp.array(-0.1, jnp.float32)
        else:
            assert timestep.reward == jnp.array(10.9, jnp.float32)


def test_boxoban__reward_function_random(boxoban_simple: Boxoban) -> None:
    """Check the reward function is correct when randomly acting in the
    trivial problem, where accidently pushing boxes onto targets is likely.
    Every step should give -0.1, each box pushed on adds 1 , each box removed
    on takes away 1 ,solving adds an additional 10"""

    def check_correct_reward(
        timestep: TimeStep,
        num_boxes_on_targets_new: chex.Array,
        num_boxes_on_targets: chex.Array,
    ) -> None:

        if num_boxes_on_targets_new == jnp.array(4, jnp.int32):
            assert timestep.reward == jnp.array(10.9, jnp.float32)
        elif num_boxes_on_targets_new - num_boxes_on_targets > jnp.array(0, jnp.int32):
            assert timestep.reward == jnp.array(0.9, jnp.float32)
        elif num_boxes_on_targets_new - num_boxes_on_targets < jnp.array(0, jnp.int32):
            assert timestep.reward == jnp.array(-1.1, jnp.float32)
        else:
            assert timestep.reward == jnp.array(-0.1, jnp.float32)

    for i in range(5):
        chex.clear_trace_counter()
        step_fn = jax.jit(chex.assert_max_traces(boxoban_simple.step, n=1))

        key = jax.random.PRNGKey(i)
        reset_key, step_key = jax.random.split(key)
        state, timestep = boxoban_simple.reset(reset_key)

        num_boxes_on_targets = boxoban_simple.count_targets(state)

        for _ in range(120):
            action = jnp.array(random.randint(0, 4), jnp.int32)
            state, timestep = step_fn(state, action)

            num_boxes_on_targets_new = boxoban_simple.count_targets(state)

            check_correct_reward(
                timestep, num_boxes_on_targets_new, num_boxes_on_targets
            )

            num_boxes_on_targets = num_boxes_on_targets_new


def test_boxoban__does_not_smoke(boxoban: Boxoban) -> None:
    """Test that we can run an episode without any errors."""
    check_env_does_not_smoke(boxoban)