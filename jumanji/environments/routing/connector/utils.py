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

from jumanji.environments.routing.connector.constants import PATH, POSITION, TARGET


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
