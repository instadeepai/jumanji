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

from jumanji.environments.routing.connector.constants import (
    AGENT_INITIAL_VALUE,
    EMPTY,
    PATH,
    POSITION,
    TARGET,
)
from jumanji.environments.routing.connector.types import Agent


def get_path(agent_id: jnp.int32) -> jnp.int32:
    """Get the value used in the state to represent the path of the given agent."""
    return PATH + 3 * agent_id


def get_position(agent_id: jnp.int32) -> jnp.int32:
    """Get the value used in the state to represent the position of the given agent."""
    return POSITION + 3 * agent_id


def get_target(agent_id: jnp.int32) -> jnp.int32:
    """Get the value used in the state to represent the target of the given agent."""
    return TARGET + 3 * agent_id


def move(position: chex.Array, action: jnp.int32) -> chex.Array:
    """Use a position and an action to return a new position.

    Args:
        position: a position representing row and column.
        action: the action representing cardinal directions.

    Returns:
        The new position after the move.
    """
    row, col = position

    move_noop = lambda row, col: jnp.array([row, col], jnp.int32)
    move_left = lambda row, col: jnp.array([row, col - 1], jnp.int32)
    move_up = lambda row, col: jnp.array([row - 1, col], jnp.int32)
    move_right = lambda row, col: jnp.array([row, col + 1], jnp.int32)
    move_down = lambda row, col: jnp.array([row + 1, col], jnp.int32)

    return jax.lax.switch(
        action, [move_noop, move_up, move_right, move_down, move_left], row, col
    )


def is_valid_position(
    grid: chex.Array, agent: Agent, position: chex.Array
) -> chex.Array:
    """Checks to see if the specified agent can move to `position`.

    Args:
        grid: the environment state's grid.
        agent: the agent.
        position: the new position for the agent.

    Returns:
        bool: True if the agent moving to position is valid.
    """
    row, col = position
    grid_size = grid.shape[0]

    # Within the bounds of the grid
    in_bounds = (0 <= row) & (row < grid_size) & (0 <= col) & (col < grid_size)
    # Cell is not occupied
    open_cell = (grid[row, col] == EMPTY) | (grid[row, col] == get_target(agent.id))
    # Agent is not connected
    not_connected = ~agent.connected

    return in_bounds & open_cell & not_connected


def get_agent_grid(agent_id: jnp.int32, grid: chex.Array) -> chex.Array:
    """Returns the grid with zeros everywhere except locations related to the desired agent:
    path, position, or target represented by 1, 2, 3 for the first agent, 4, 5, 6 for the
    second agent, etc."""
    position = get_position(agent_id)
    target = get_target(agent_id)
    path = get_path(agent_id)
    agent_head = (grid == position) * position
    agent_target = (grid == target) * target
    agent_path = (grid == path) * path
    return agent_head + agent_target + agent_path


def switch_perspective(grid: chex.Array, agent_id: int, num_agents: int) -> chex.Array:
    """Encodes the observation with respect to the current agent defined by `agent_id`.

    `agent_id`'s observations will always take the values 1,2,3 and ordering of observations
    will be preserved (agent 0's observations always come before agent 1's).

    For example, in a 3-agent game, if we wanted to switch to
    agent 1's perspective, then:
    agent 1s values will change from 4,5,6 -> 1,2,3
    agent 2s values will change from 7,8,9 -> 4,5,6
    agent 0s values will change from 1,2,3 -> 7,8,9

    Agent 0 will be passed observations where it is represented by the values 1,2,3. Agent 1
    will be passed observations where it is represented by the values 1,2,3. However in the
    state agent 0 will always be 1,2,3 and agent 1 will always be 4,5,6.

    Args:
        grid: the environment state grid.
        agent_id: the id of the agent whose observation is being requested.
        num_agents: the number of agents on the grid.

    Returns:
        Array: the grid in the perspective of the given agent id.
    """
    new_grid = grid - AGENT_INITIAL_VALUE  # Center agent values around 0
    new_grid -= 3 * agent_id  # Move the obs
    new_grid %= 3 * num_agents  # Keep obs in bounds
    new_grid += AGENT_INITIAL_VALUE  # 'Un-center' agent obs around 0
    # Take agent values from rotated grid and empty values from old grid
    return jnp.where((grid >= AGENT_INITIAL_VALUE), new_grid, grid)
