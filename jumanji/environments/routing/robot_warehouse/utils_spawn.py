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

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.robot_warehouse.constants import (
    _AGENTS,
    _POSSIBLE_DIRECTIONS,
    _SHELVES,
)
from jumanji.environments.routing.robot_warehouse.types import (
    Agent,
    Entity,
    Position,
    Shelf,
)
from jumanji.environments.routing.robot_warehouse.utils import get_entity_ids
from jumanji.tree_utils import tree_slice


def spawn_agent(
    agent_coordinates: chex.Array,
    direction: chex.Array,
) -> chex.Array:
    """Spawn an agent (robot) at a given position and direction.

    Args:
        agent_coordinates: x, y coordinates of the agent.
        direction: direction of the agent.

    Returns:
        spawned agent.
    """
    x, y = agent_coordinates
    agent_pos = Position(x=x, y=y)
    agent = Agent(position=agent_pos, direction=direction, is_carrying=0)
    return agent


def spawn_shelf(
    shelf_coordinates: chex.Array,
    requested: chex.Array,
) -> chex.Array:
    """Spawn a shelf at a specific shelf position and label the shelf
    as requested or not.

    Args:
        shelf_coordinates: x, y coordinates of the shelf.
        requested: whether the shelf has been requested or not.

    Returns:
        spawned shelf.
    """
    x, y = shelf_coordinates
    shelf_pos = Position(x=x, y=y)
    shelf = Shelf(position=shelf_pos, is_requested=requested)
    return shelf


def spawn_random_entities(
    key: chex.PRNGKey,
    grid_size: chex.Array,
    agent_ids: chex.Array,
    shelf_ids: chex.Array,
    shelf_coordinates: chex.Array,
    request_queue_size: chex.Array,
) -> Tuple[chex.PRNGKey, Agent, Shelf, chex.Array]:
    """Spawn agents and shelves on the warehouse floor grid.

    Args:
        key: pseudo random number key.
        grid_size: the size of the warehouse floor grid.
        agent_ids: array of agent ids.
        shelf_ids: array of shelf ids.
        shelf_coordinates: x,y coordinates of shelf positions.
        request_queue_size: the number of shelves to be delivered.

    Returns:
        new key, spawned agents, shelves and the request queue.
    """

    # random agent positions
    num_agents = len(agent_ids)
    key, position_key = jax.random.split(key)
    grid_cells = jnp.array(jnp.arange(grid_size[0] * grid_size[1]))
    agent_coords = jax.random.choice(
        position_key,
        grid_cells,
        shape=(num_agents,),
        replace=False,
    )
    agent_coords = jnp.transpose(
        jnp.asarray(jnp.unravel_index(agent_coords, grid_size))
    )

    # random agent directions
    key, direction_key = jax.random.split(key)

    agent_dirs = jax.random.choice(
        direction_key, _POSSIBLE_DIRECTIONS, shape=(num_agents,)
    )

    # sample request queue
    key, queue_key = jax.random.split(key)
    shelf_request_queue = jax.random.choice(
        queue_key,
        shelf_ids,
        shape=(request_queue_size,),
        replace=False,
    )
    requested_ids = jnp.zeros(shelf_ids.shape)
    requested_ids = requested_ids.at[shelf_request_queue].set(1)

    # spawn agents and shelves
    agents = jax.vmap(spawn_agent)(agent_coords, agent_dirs)
    shelves = jax.vmap(spawn_shelf)(shelf_coordinates, requested_ids)
    return key, agents, shelves, shelf_request_queue


def place_entity_on_grid(
    grid: chex.Array,
    channel: chex.Array,
    entities: Entity,
    entity_id: chex.Array,
) -> chex.Array:
    """Places an entity (Agent/Shelf) on the grid based on its
    (x, y) position defined once spawned.

    Args:
        grid: the warehouse floor grid array.
        channel: the grid channel index, either agents or shelves.
        entities: a pytree of Agent or Shelf type containing entity information.
        entity_id: unique ID identifying a specific entity.

    Returns:
        the warehouse grid with the specific entity in its position.
    """
    entity = tree_slice(entities, entity_id)
    x, y = entity.position.x, entity.position.y
    return grid.at[channel, x, y].set(entity_id + 1)


def place_entities_on_grid(
    grid: chex.Array, agents: Agent, shelves: Shelf
) -> chex.Array:
    """Place agents and shelves on the grid.

    Args:
        grid: the warehouse floor grid array.
        agents: a pytree of Agent type containing agent information.
        shelves: a pytree of Shelf type containing shelf information.

    Returns:
        the warehouse grid with all agents and shelves placed in their
        positions.
    """
    agent_ids = get_entity_ids(agents)
    shelf_ids = get_entity_ids(shelves)

    # place agents and shelves on warehouse grid
    def place_agents_scan(
        grid_and_agents: Tuple[chex.Array, chex.Array], agent_id: chex.Array
    ) -> Tuple[Tuple[chex.Array, chex.Array], None]:
        grid, agents = grid_and_agents
        grid = place_entity_on_grid(grid, _AGENTS, agents, agent_id)
        return (grid, agents), None

    def place_shelves_scan(
        grid_and_shelves: Tuple[chex.Array, chex.Array], shelf_id: chex.Array
    ) -> Tuple[Tuple[chex.Array, chex.Array], None]:
        grid, shelves = grid_and_shelves
        grid = place_entity_on_grid(grid, _SHELVES, shelves, shelf_id)
        return (grid, shelves), None

    (grid, _), _ = jax.lax.scan(place_agents_scan, (grid, agents), agent_ids)
    (grid, _), _ = jax.lax.scan(place_shelves_scan, (grid, shelves), shelf_ids)
    return grid
