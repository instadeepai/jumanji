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

from typing import Tuple

import jax
import jax.numpy as jnp
import pytest
from jax import random

from jumanji.routing import Routing
from jumanji.routing.types import Position, State
from jumanji.testing.env_not_smoke import check_env_does_not_smoke
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import TimeStep


@pytest.fixture
def routing_env() -> Routing:
    """Instantiates a default Routing environment."""
    return Routing(12, 12, 2)


@pytest.fixture
def deterministic_routing_env() -> Tuple[Routing, State, TimeStep]:
    """Instantiates a 3x3 Routing environment with a step limit of 5."""
    env = Routing(3, 3, 2, step_limit=5)
    state, timestep, _ = env.reset(random.PRNGKey(10))
    state.grid = jnp.array(
        [
            [4, 0, 0],
            [3, 0, 0],
            [0, 7, 6],
        ]
    )
    return env, state, timestep


def test_routing__reset(routing_env: Routing) -> None:
    """Validates the jitted reset of the environment."""
    reset_fn = jax.jit(routing_env.reset)
    key1, key2 = random.PRNGKey(0), random.PRNGKey(1)
    state1, timestep1, _ = reset_fn(key1)
    state2, timestep2, _ = reset_fn(key2)
    assert isinstance(timestep1, TimeStep)
    assert isinstance(state1, State)
    assert state1.step == 0
    assert state1.grid.shape == (routing_env.rows, routing_env.cols)
    assert jnp.all(state1.grid == timestep1.observation[0])
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # reset function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(state1)
    # Check random initialization
    assert not jnp.all(state1.key == state2.key)
    assert not jnp.all(state1.grid == state2.grid)
    assert jnp.all(state1.finished_agents == state2.finished_agents)
    assert state1.step == state2.step


def test_routing__finished_agents_behaviour(routing_env: Routing) -> None:
    """Validates the behaviour of finished_agents variable"""
    grid = jnp.array([[3, 0, 4, 2], [0, 0, 0, 2], [0, 0, 0, 0], [6, 0, 7, 5]])
    state = State(
        key=random.PRNGKey(0),
        grid=grid,
        step=0,
        finished_agents=jnp.array([False, False]),
    )

    state, _, _ = routing_env.step(state, jnp.array([1, 1]))

    state2, _, _ = routing_env.step(state, jnp.array([1, 0]))
    state3, _, _ = routing_env.step(state, jnp.array([0, 1]))
    state4, _, _ = routing_env.step(state, jnp.array([1, 1]))

    assert jnp.all(state2.finished_agents == jnp.array([True, False]))
    assert jnp.all(state3.finished_agents == jnp.array([False, True]))
    assert jnp.all(state4.finished_agents == jnp.array([True, True]))


def test_routing__agent_observation(routing_env: Routing) -> None:
    """Validates the agent observation function."""
    grid = jnp.array([[3, 0, 4, 2], [0, 0, 0, 2], [0, 0, 0, 0], [6, 0, 7, 5]])
    grid2 = jnp.array([[6, 0, 7, 5], [0, 0, 0, 5], [0, 0, 0, 0], [3, 0, 4, 2]])

    state1 = State(
        key=random.PRNGKey(0),
        grid=grid,
        step=0,
        finished_agents=jnp.array([False, False]),
    )

    state2, timestep, _ = routing_env.step(state1, jnp.array([0, 0]))

    assert jnp.all(timestep.observation[0] == grid)

    assert jnp.all(timestep.observation[1] == grid2)

    test_env = Routing(9, 9, 6)

    grid = jnp.array(
        [
            [3, 0, 4, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [6, 0, 7, 5, 0, 0, 0, 0, 0],
            [8, 10, 0, 9, 0, 0, 0, 0, 0],
            [0, 11, 13, 12, 0, 0, 0, 0, 0],
            [15, 16, 14, 0, 0, 0, 0, 0, 0],
            [17, 19, 18, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    state1 = State(
        key=random.PRNGKey(0),
        grid=grid,
        step=0,
        finished_agents=jnp.array([False, False, False, False, False, False]),
    )

    state2, timestep, _ = test_env.step(state1, jnp.array([0, 0, 0, 0, 0, 0]))

    assert jnp.all(timestep.observation[0] == grid)

    grid2 = jnp.array(
        [
            [18, 0, 19, 17, 0, 0, 0, 0, 0],
            [0, 0, 0, 17, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 0, 4, 2, 0, 0, 0, 0, 0],
            [8 - 3, 10 - 3, 0, 9 - 3, 0, 0, 0, 0, 0],
            [0, 11 - 3, 13 - 3, 12 - 3, 0, 0, 0, 0, 0],
            [15 - 3, 16 - 3, 14 - 3, 0, 0, 0, 0, 0, 0],
            [17 - 3, 19 - 3, 18 - 3, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    assert jnp.all(timestep.observation[1] == grid2)

    grid3 = jnp.array(
        [
            [15, 0, 16, 14, 0, 0, 0, 0, 0],
            [0, 0, 0, 14, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [18, 0, 19, 17, 0, 0, 0, 0, 0],
            [2, 4, 0, 3, 0, 0, 0, 0, 0],
            [0, 11 - 6, 13 - 6, 12 - 6, 0, 0, 0, 0, 0],
            [15 - 6, 16 - 6, 14 - 6, 0, 0, 0, 0, 0, 0],
            [17 - 6, 19 - 6, 18 - 6, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    assert jnp.all(timestep.observation[2] == grid3)


def test_routing__step(routing_env: Routing) -> None:
    """Validates the jitted step function of the environment."""
    step_fn = jax.jit(routing_env.step)
    state_key, action_key1, action_key2 = random.split(random.PRNGKey(10), 3)
    state, timestep, _ = routing_env.reset(state_key)

    # Sample two different actions
    action1, action2 = random.choice(
        key=action_key1,
        a=jnp.arange(5),
        shape=(2,),
        replace=False,
        p=routing_env.get_action_mask(state.grid, 0),
    )

    action1 = jnp.zeros((routing_env.num_agents,), int).at[0].set(action1)
    action2 = jnp.zeros((routing_env.num_agents,), int).at[0].set(action2)

    new_state1, timestep1, _ = step_fn(state, action1)

    # Check that rewards are within the correct range
    assert jnp.all(timestep1.reward <= routing_env._reward_connected)
    assert jnp.all(timestep1.reward >= routing_env._reward_blocked)

    # Check that rewards have the correct number of dimensions
    assert jnp.ndim(timestep1.reward) == 1
    assert len(timestep.reward == routing_env.num_agents)
    # Check that discounts have the correct number of dimensions
    assert jnp.ndim(timestep1.discount) == 1
    assert len(timestep.discount == routing_env.num_agents)
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # step function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(new_state1)
    # Check that the state has changed
    assert new_state1.step != state.step
    assert not jnp.all(new_state1.grid != state.grid)
    # Check that two different actions lead to two different states
    new_state2, timestep2, _ = step_fn(state, action2)
    assert not jnp.all(new_state1.grid != new_state2.grid)
    # Check that the state update and timestep creation work as expected
    head, _ = routing_env._extract_agent_information(state.grid, 0)
    row, col = head

    moves = {
        1: (Position(x=row, y=col - 1)),  # Left
        2: (Position(x=row - 1, y=col)),  # Up
        3: (Position(x=row, y=col + 1)),  # Right
        4: (Position(x=row + 1, y=col)),  # Down
    }
    for action, new_position in moves.items():
        new_state, timestep, _ = step_fn(state, jnp.array([action, action]))
        if routing_env._is_valid(state.grid, 0, new_position):
            head, _ = routing_env._extract_agent_information(new_state.grid, 0)
            posx, posy = head
            targx, targy = new_position
            assert (posx == targx) & (posy == targy)


def test_routing__does_not_smoke(routing_env: Routing) -> None:
    """Test that we can run an episode without any errors."""
    check_env_does_not_smoke(routing_env)


def test_routing__step_limit(routing_env: Routing) -> None:
    """Validates the terminal reward."""
    step_fn = jax.jit(routing_env.step)
    state_key, action_key1, action_key2 = random.split(random.PRNGKey(10), 3)
    state, timestep, _ = routing_env.reset(state_key)

    for _ in range(routing_env._step_limit):
        state, timestep, _ = step_fn(state, jnp.array([0, 0]))

    assert timestep.mid()
    reward = timestep.reward
    state, timestep, _ = step_fn(state, jnp.array([0, 0]))
    new_reward = timestep.reward
    assert timestep.last()
    assert jnp.all(new_reward == (reward + routing_env._reward_for_terminal_step))


def test__routing__termination(
    deterministic_routing_env: Tuple[Routing, State, TimeStep]
) -> None:
    routing_env, state, timestep = deterministic_routing_env
    step_fn = jax.jit(routing_env.step)

    # termination
    # agent 0 connects and agent 1 is blocked
    state, timestep, _ = step_fn(state, jnp.array([4, 1]))
    assert timestep.last()
    assert jnp.all(timestep.discount == 0)


def test__routing__truncation(
    deterministic_routing_env: Tuple[Routing, State, TimeStep]
) -> None:
    routing_env, state, timestep = deterministic_routing_env
    step_fn = jax.jit(routing_env.step)

    # truncation
    for _ in range(routing_env._step_limit + 1):
        state, timestep, _ = step_fn(state, jnp.array([0, 0]))

    assert timestep.last()
    assert not jnp.all(timestep.discount == 0)


def test__routing__termination_at_horizon(
    deterministic_routing_env: Tuple[Routing, State, TimeStep]
) -> None:
    routing_env, state, timestep = deterministic_routing_env
    step_fn = jax.jit(routing_env.step)

    # termination at final step and no truncation
    for _ in range(routing_env._step_limit):
        state, timestep, _ = step_fn(state, jnp.array([0, 0]))

        # no termination yet
        assert timestep.mid()
        assert not jnp.all(timestep.discount == 0)

    # time horizon and all complete
    # both agents connect
    state, timestep, _ = step_fn(state, jnp.array([4, 3]))
    assert timestep.last()
    assert jnp.all(timestep.discount == 0)


def test_routing__action_masking(routing_env: Routing) -> None:
    """Validates the action masking."""
    grid = jnp.array(
        [
            [3, 0, 4, 2, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [6, 0, 7, 5, 0],
        ]
    )

    state1 = State(
        key=random.PRNGKey(0),
        grid=grid,
        step=0,
        finished_agents=jnp.array([False, False]),
    )

    agent_0_mask = routing_env.get_action_mask(state1.grid, 0)
    agent_1_mask = routing_env.get_action_mask(state1.grid, 1)

    assert jnp.all(agent_0_mask == jnp.array([1, 1, 0, 0, 1]))
    assert jnp.all(agent_1_mask == jnp.array([1, 1, 1, 0, 0]))

    state2, timestep, _ = routing_env.step(state1, jnp.array([1, 1]))

    agent_0_mask = routing_env.get_action_mask(state2.grid, 0)
    agent_1_mask = routing_env.get_action_mask(state2.grid, 1)

    assert jnp.all(agent_0_mask == jnp.array([1, 1, 0, 0, 1]))
    assert jnp.all(agent_1_mask == jnp.array([1, 1, 1, 0, 0]))

    state3, timestep, _ = routing_env.step(state2, jnp.array([4, 2]))

    agent_0_mask = routing_env.get_action_mask(state3.grid, 0)
    agent_1_mask = routing_env.get_action_mask(state3.grid, 1)

    assert jnp.all(agent_0_mask == jnp.array([1, 1, 0, 1, 1]))
    assert jnp.all(agent_1_mask == jnp.array([1, 1, 1, 1, 0]))

    # Test that the function works correctly using other observations
    agent_0_mask = routing_env.get_action_mask(timestep.observation[1], 0)
    agent_1_mask = routing_env.get_action_mask(timestep.observation[1], 1)

    assert jnp.all(agent_1_mask == jnp.array([1, 1, 0, 1, 1]))
    assert jnp.all(agent_0_mask == jnp.array([1, 1, 1, 1, 0]))

    state4, timestep, _ = routing_env.step(state3, jnp.array([3, 3]))

    agent_0_mask = routing_env.get_action_mask(state4.grid, 0)
    agent_1_mask = routing_env.get_action_mask(state4.grid, 1)

    assert jnp.all(agent_0_mask == jnp.array([1, 0, 0, 0, 1]))
    assert jnp.all(agent_1_mask == jnp.array([1, 0, 1, 1, 0]))
