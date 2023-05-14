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
import jax.numpy as jnp
import pytest

from jumanji.environments.routing.connector.constants import (
    DOWN,
    EMPTY,
    LEFT,
    NOOP,
    RIGHT,
    UP,
)
from jumanji.environments.routing.connector.types import Agent, State
from jumanji.environments.routing.connector.utils import (
    connected_or_blocked,
    get_agent_grid,
    get_path,
    get_position,
    get_target,
    is_valid_position,
    move_agent,
    move_position,
)
from jumanji.tree_utils import tree_slice


def test_get_path() -> None:
    """Tests that get trace only returns traces."""
    assert get_path(0) == 1
    assert get_path(1) == 4
    assert get_path(5) == 16


def test_get_head() -> None:
    """Tests that get head only returns heads."""
    assert get_position(0) == 2
    assert get_position(1) == 5
    assert get_position(5) == 17


def test_get_target() -> None:
    """Tests that get target only returns targets."""
    assert get_target(0) == 3
    assert get_target(1) == 6
    assert get_target(5) == 18


def test_move_position() -> None:
    """Test that move returns the correct tuple for the correct type of move."""
    pos = jnp.array([1, 1])
    assert (move_position(pos, NOOP) == jnp.array([1, 1])).all()
    assert (move_position(pos, UP) == jnp.array([0, 1])).all()
    assert (move_position(pos, RIGHT) == jnp.array([1, 2])).all()
    assert (move_position(pos, DOWN) == jnp.array([2, 1])).all()
    assert (move_position(pos, LEFT) == jnp.array([1, 0])).all()


def test_move_agent(state: State) -> None:
    """Tests that `move_agent` returns the correct Agent struct."""
    new_position = jnp.array([1, 1])
    agent0 = tree_slice(state.agents, 0)

    new_agent, grid = move_agent(agent0, state.grid, new_position)

    assert (new_agent.position == new_position).all()
    assert grid[tuple(new_agent.position)] == get_position(0)
    assert grid[tuple(agent0.position)] == get_path(0)


def test_move_agent_invalid(state: State) -> None:
    """Tests that `move_agent` throws an error when invalid array is passed."""
    invalid_position = jnp.array([1, 1, 1])
    agent0 = tree_slice(state.agents, 0)

    with pytest.raises(IndexError):
        move_agent(agent0, state.grid, invalid_position)


def test_is_valid_position(state: State) -> None:
    """Tests that the _is_valid_move method flags invalid moves."""
    agent1 = tree_slice(state.agents, 1)
    valid_move = is_valid_position(
        grid=state.grid, agent=agent1, position=jnp.array([2, 2])
    )
    move_into_path = is_valid_position(
        grid=state.grid, agent=agent1, position=jnp.array([4, 2])
    )

    assert valid_move
    assert not move_into_path


def test_connected_or_blocked() -> None:
    """Tests that connected or blocked only returns false when an agent
    is neither connected nor blocked.
    """
    not_connected_agent = Agent(
        id=jnp.array(0, jnp.int32),
        start=jnp.array([1, 1]),
        target=jnp.array([1, 3]),
        position=jnp.array([1, 2]),
    )
    connected_agent = Agent(
        id=jnp.array(0, jnp.int32),
        start=jnp.array([1, 2]),
        target=jnp.array([1, 2]),
        position=jnp.array([1, 2]),
    )
    not_connected_not_blocked = connected_or_blocked(not_connected_agent, jnp.ones(5))
    connected_and_blocked = connected_or_blocked(connected_agent, jnp.zeros(5))
    not_connected_blocked = connected_or_blocked(not_connected_agent, jnp.zeros(5))
    connected_not_blocked = connected_or_blocked(connected_agent, jnp.ones(5))

    assert not not_connected_not_blocked
    assert connected_and_blocked
    assert not_connected_blocked
    assert connected_not_blocked


def test_get_agent_grid(grid: chex.Array) -> None:
    """Test that the agent grid only contains items related to a single agent."""
    agent_0_grid = get_agent_grid(0, grid)
    agent_1_grid = get_agent_grid(1, grid)
    agent_2_grid = get_agent_grid(2, grid)

    assert jnp.all(agent_0_grid <= 4)
    assert jnp.all(
        (agent_1_grid == EMPTY)
        | (agent_1_grid == get_path(1))
        | (agent_1_grid == get_target(1))
        | (agent_1_grid == get_position(1))
    )
    assert jnp.all(
        (agent_2_grid == EMPTY)
        | (agent_2_grid == get_path(2))
        | (agent_2_grid == get_target(2))
        | (agent_2_grid == get_position(2))
    )
