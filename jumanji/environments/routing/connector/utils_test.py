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
import pytest

from jumanji.environments.routing.connector.constants import (
    DOWN,
    EMPTY,
    LEFT,
    NOOP,
    PATH,
    RIGHT,
    UP,
)
from jumanji.environments.routing.connector.types import Agent, State
from jumanji.environments.routing.connector.utils import (
    connected_or_blocked,
    get_action_masks,
    get_adjacency_mask,
    get_agent_grid,
    get_path,
    get_position,
    get_surrounded_mask,
    get_target,
    is_repeated_later,
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
        value_on_grid=0, agent=agent1, position=jnp.array([2, 2]), grid_size=state.grid.shape[0]
    )
    move_into_path = is_valid_position(
        value_on_grid=PATH,
        agent=agent1,
        position=jnp.array([4, 2]),
        grid_size=state.grid.shape[0],
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


def test_is_repeated_later() -> None:
    """Test that the method is_repeated_later finds future duplicated in a list of positions"""

    repeated_later_jit = jax.jit(is_repeated_later)

    # Example 1: simple
    positions1 = jnp.array([[1, 1], [2, 2], [1, 1], [3, 3], [2, 2]])
    mask1 = repeated_later_jit(positions1)
    assert jnp.array_equal(mask1, jnp.array([True, True, False, False, False]))

    # Example 2: All unique
    positions2 = jnp.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    mask2 = repeated_later_jit(positions2)
    assert jnp.array_equal(mask2, jnp.array([False, False, False, False]))

    # Example 3: All same
    positions3 = jnp.array([[5, 5], [5, 5], [5, 5]])
    mask3 = repeated_later_jit(positions3)
    assert jnp.array_equal(mask3, jnp.array([True, True, False]))

    # Example 4: Empty list of positions
    positions4 = jnp.array([]).reshape(0, 2)  # or jnp.empty((0,2), dtype=int)
    mask4 = repeated_later_jit(positions4)
    assert jnp.array_equal(mask4, jnp.array([]))

    # Example 5: Single position
    positions5 = jnp.array([[10, 20]])
    mask5 = repeated_later_jit(positions5)
    assert jnp.array_equal(mask5, jnp.array([False]))

    # Example 6: complex
    positions6 = jnp.array(
        [[1, 1], [2, 2], [3, 3], [4, 4], [1, 1], [3, 3], [2, 2], [5, 5], [6, 6], [1, 1], [3, 3]]
    )
    mask6 = repeated_later_jit(positions6)
    assert jnp.array_equal(
        mask6, jnp.array([True, True, True, False, True, True, False, False, False, False, False])
    )


def test_get_action_masks(state: State) -> None:
    """Validates the action masking."""
    action_masks1 = get_action_masks(state.agents, state.grid)
    expected_mask = jnp.array(
        [
            [True, True, False, True, True],
            [True, True, True, False, True],
            [True, False, True, False, True],
        ]
    )

    assert jnp.array_equal(action_masks1, expected_mask)


def test_get_surrounded_mask() -> None:
    """Tests that the get_surrounded_mask function returns the correct mask."""
    grid = jnp.array(
        [
            [0, 1, 2, 0, 0],
            [10, 0, 4, 0, 0],
            [11, 7, 5, 0, 0],
            [0, 8, 0, 0, 0],
        ]
    )

    expected = jnp.array(
        [
            [True, False, False, False, False],
            [False, True, False, False, False],
            [False, False, False, False, False],
            [True, False, False, False, False],
        ]
    )

    assert jnp.array_equal(get_surrounded_mask(grid), expected)


def test_get_adjacency_mask() -> None:
    """Tests that the get_adjacency_mask function returns the correct mask."""
    grid_shape = (4, 4)
    coordinate = jnp.array([2, 3])
    expected = jnp.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=jnp.bool_,
    )

    actual = get_adjacency_mask(grid_shape, coordinate)

    assert jnp.array_equal(actual, expected)
