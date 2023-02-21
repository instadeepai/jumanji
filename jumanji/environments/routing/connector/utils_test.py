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

from jumanji.environments.routing.connector.constants import (
    DOWN,
    EMPTY,
    LEFT,
    NOOP,
    RIGHT,
    UP,
)
from jumanji.environments.routing.connector.types import State
from jumanji.environments.routing.connector.utils import (
    get_agent_grid,
    get_path,
    get_position,
    get_target,
    is_valid_position,
    move,
    switch_perspective,
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


def test_move() -> None:
    """Test that move returns the correct tuple for the correct type of move."""
    pos = jnp.array([1, 1])
    assert (move(pos, NOOP) == jnp.array([1, 1])).all()
    assert (move(pos, UP) == jnp.array([0, 1])).all()
    assert (move(pos, RIGHT) == jnp.array([1, 2])).all()
    assert (move(pos, DOWN) == jnp.array([2, 1])).all()
    assert (move(pos, LEFT) == jnp.array([1, 0])).all()


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


def test_switch_perspective(grid: chex.Array) -> None:
    """Tests that observations are correctly generated given the grid."""
    observations0 = switch_perspective(grid, agent_id=0, num_agents=3)
    observations1 = switch_perspective(grid, agent_id=1, num_agents=3)
    observations2 = switch_perspective(grid, agent_id=2, num_agents=3)

    path0 = get_path(0)
    path1 = get_path(1)
    path2 = get_path(2)

    targ0 = get_target(0)
    targ1 = get_target(1)
    targ2 = get_target(2)

    posi0 = get_position(0)
    posi1 = get_position(1)
    posi2 = get_position(2)

    empty = EMPTY

    expected_agent_1 = jnp.array(
        [
            [empty, empty, targ2, empty, empty, empty],
            [empty, empty, posi2, path2, path2, empty],
            [empty, empty, empty, targ1, posi1, empty],
            [targ0, empty, posi0, empty, path1, empty],
            [empty, empty, path0, empty, path1, empty],
            [empty, empty, path0, empty, empty, empty],
        ]
    )
    expected_agent_2 = jnp.array(
        [
            [empty, empty, targ1, empty, empty, empty],
            [empty, empty, posi1, path1, path1, empty],
            [empty, empty, empty, targ0, posi0, empty],
            [targ2, empty, posi2, empty, path0, empty],
            [empty, empty, path2, empty, path0, empty],
            [empty, empty, path2, empty, empty, empty],
        ]
    )

    assert (grid == observations0).all()
    assert (expected_agent_1 == observations1).all()
    assert (expected_agent_2 == observations2).all()
