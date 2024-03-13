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

from functools import partial

import chex
import jax
import jax.numpy as jnp

import jumanji.testing.pytrees
from jumanji.environments.routing.connector import constants
from jumanji.environments.routing.connector.constants import EMPTY
from jumanji.environments.routing.connector.env import Connector
from jumanji.environments.routing.connector.types import Agent, State
from jumanji.environments.routing.connector.utils import get_position, get_target
from jumanji.testing.env_not_smoke import (
    check_env_does_not_smoke,
    check_env_specs_does_not_smoke,
)
from jumanji.tree_utils import tree_slice
from jumanji.types import StepType, TimeStep


@partial(jax.vmap, in_axes=(0, None))
def is_head_on_grid(agent: Agent, grid: chex.Array) -> chex.Array:
    """Returns true if the agent's head is on the correct place on the grid."""
    return jnp.any(grid[agent.position] == get_position(agent.id))


@partial(jax.vmap, in_axes=(0, None))
def is_target_on_grid(agent: Agent, grid: chex.Array) -> chex.Array:
    """Returns true if the agent's target is on the correct place on the grid."""
    return jnp.any(grid[agent.target] == get_target(agent.id))


def test_connector__reset(connector: Connector, key: jax.random.PRNGKey) -> None:
    """Test that all heads and targets are on the board."""
    state, timestep = connector.reset(key)

    assert state.grid.shape == (connector.grid_size, connector.grid_size)

    for agent_id in range(connector.num_agents):
        assert jnp.any(state.grid == get_position(agent_id))
        assert jnp.any(state.grid == get_target(agent_id))

    assert all(is_head_on_grid(state.agents, state.grid))
    assert all(is_target_on_grid(state.agents, state.grid))

    assert timestep.discount == 1.0
    assert timestep.reward == 0.0
    assert timestep.step_type == StepType.FIRST


def test_connector__reset_jit(connector: Connector) -> None:
    """Confirm that the reset is only compiled once when jitted."""
    chex.clear_trace_counter()
    reset_fn = jax.jit(chex.assert_max_traces(connector.reset, n=1))
    key = jax.random.PRNGKey(0)
    _ = reset_fn(key)

    # Call again to check it does not compile twice
    state, timestep = reset_fn(key)
    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)


def test_connector__step_connected(
    connector: Connector,
    state: State,
    state1: State,
    state2: State,
    action1: chex.Array,
    action2: chex.Array,
) -> None:
    """Tests that timestep is done when all agents connect"""
    step_fn = jax.jit(connector.step)
    real_state1, timestep = step_fn(state, action1)
    reward = connector._reward_fn(state, action1, real_state1)
    assert jnp.array_equal(timestep.reward, reward)
    chex.assert_trees_all_equal(real_state1, state1)

    real_state2, timestep = step_fn(real_state1, action2)
    chex.assert_trees_all_equal(real_state2, state2)

    assert timestep.step_type == StepType.LAST
    assert jnp.array_equal(timestep.discount, jnp.asarray(0))
    reward = connector._reward_fn(real_state1, action2, real_state2)
    assert jnp.array_equal(timestep.reward, reward)

    assert all(is_head_on_grid(state.agents, state.grid))
    # None of the targets should be on the grid because everyone is connected
    assert not any(is_target_on_grid(real_state2.agents, real_state2.grid))


def test_connector__step_blocked(
    connector: Connector,
    state: State,
    path0: int,
    path1: int,
    path2: int,
    targ0: int,
    targ1: int,
    targ2: int,
    posi0: int,
    posi1: int,
    posi2: int,
) -> None:
    """Tests that timestep is done when all agents are blocked"""
    step_fn = jax.jit(connector.step)
    # Actions that will block all agents
    actions = jnp.array(
        [
            [constants.LEFT, constants.LEFT, constants.RIGHT],
            [constants.DOWN, constants.DOWN, constants.UP],
            [constants.RIGHT, constants.LEFT, constants.UP],
            [constants.NOOP, constants.DOWN, constants.LEFT],
            [constants.NOOP, constants.RIGHT, constants.LEFT],
        ]
    )

    # Take the actions that will block all agents
    for action in actions:
        state, timestep = step_fn(state, action)

    expected_grid = jnp.array(
        [
            [EMPTY, EMPTY, targ0, posi2, path2, path2],
            [EMPTY, path0, path0, path0, path0, path2],
            [EMPTY, path0, posi0, targ2, path2, path2],
            [targ1, path1, path1, EMPTY, path2, EMPTY],
            [path1, path1, path1, EMPTY, path2, EMPTY],
            [path1, posi1, path1, EMPTY, EMPTY, EMPTY],
        ]
    )

    assert jnp.array_equal(state.grid, expected_grid)
    assert timestep.step_type == StepType.LAST
    assert jnp.array_equal(timestep.discount, jnp.asarray(0))

    assert all(is_head_on_grid(state.agents, state.grid))
    assert all(is_target_on_grid(state.agents, state.grid))


def test_connector__step_horizon(connector: Connector, state: State) -> None:
    """Tests that the timestep is done, but discounts are not all 0 past"""
    step_fn = jax.jit(connector.step)
    # The environment has a time limit of 5.
    actions = jnp.zeros(3, int)
    # step 1
    state, timestep = step_fn(state, actions)

    # step 2, 3, 4
    for _ in range(3):
        state, timestep = step_fn(state, actions)

        assert timestep.step_type != StepType.LAST
        assert jnp.array_equal(timestep.discount, jnp.asarray(1))

    # step 5
    state, timestep = step_fn(state, actions)
    assert timestep.step_type == StepType.LAST
    assert jnp.array_equal(timestep.discount, jnp.asarray(0))


def test_connector__step_agents_collision(
    connector: Connector,
    state: State,
    path0: int,
    path1: int,
    path2: int,
    targ0: int,
    targ1: int,
    targ2: int,
    posi0: int,
    posi1: int,
    posi2: int,
) -> None:
    """Tests _step_agents function when there is a collision."""
    action = jnp.array([constants.DOWN, constants.UP, constants.NOOP])
    agents, grid = connector._step_agents(state, action)

    expected_grid = jnp.array(
        [
            [EMPTY, EMPTY, targ0, EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, posi0, path0, path0, EMPTY],
            [EMPTY, EMPTY, posi1, targ2, posi2, EMPTY],
            [targ1, EMPTY, path1, EMPTY, path2, EMPTY],
            [EMPTY, EMPTY, path1, EMPTY, path2, EMPTY],
            [EMPTY, EMPTY, path1, EMPTY, EMPTY, EMPTY],
        ]
    )

    assert jnp.array_equal(grid, expected_grid)
    assert all(is_target_on_grid(agents, grid))
    assert all(is_head_on_grid(agents, grid))


def test_connector__step_agent_valid(connector: Connector, state: State) -> None:
    """Test _step_agent method given valid position."""
    agent0 = tree_slice(state.agents, 0)
    agent, grid = connector._step_agent(agent0, state.grid, constants.LEFT)

    assert jnp.array_equal(agent.position, jnp.array([1, 1]))
    # agent should have moved
    assert jnp.any(agent.position != agent0.position)
    assert grid[1, 1] == get_position(0)


def test_connector__step_agent_invalid(connector: Connector, state: State) -> None:
    """Test _step_agent method given invalid position."""
    agent0 = tree_slice(state.agents, 0)
    agent, grid = connector._step_agent(agent0, state.grid, constants.RIGHT)

    assert jnp.array_equal(agent.position, jnp.array([1, 2]))
    # agent should not have moved
    assert jnp.array_equal(agent.position, agent0.position)
    assert grid[1, 2] == get_position(0)


def test_connector__does_not_smoke(connector: Connector) -> None:
    """Test that we can run an episode without any errors."""
    check_env_does_not_smoke(connector)


def test_connector__specs_does_not_smoke(connector: Connector) -> None:
    """Test that we can access specs without any errors."""
    check_env_specs_does_not_smoke(connector)


def test_connector__get_action_mask(state: State, connector: Connector) -> None:
    """Validates the action masking."""
    action_masks = jax.vmap(connector._get_action_mask, (0, None))(
        state.agents, state.grid
    )
    expected_mask = jnp.array(
        [
            [True, True, False, True, True],
            [True, True, True, False, True],
            [True, False, True, False, True],
        ]
    )
    assert jnp.array_equal(action_masks, expected_mask)


def test_connector__get_extras(state: State, connector: Connector) -> None:
    """Validates the `_get_extras` method to generates extras metrics."""
    extras = connector._get_extras(state)
    expected_keys = ["total_path_length", "ratio_connections", "num_connections"]
    assert set(extras.keys()) == set(expected_keys)
    jumanji.testing.pytrees.assert_is_jax_array_tree(extras)
    no_op_action = jnp.full(connector.num_agents, 0, jnp.int32)
    _, timestep = connector.step(state, no_op_action)
    assert timestep.extras == extras
