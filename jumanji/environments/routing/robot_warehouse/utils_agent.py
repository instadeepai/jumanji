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

from typing import Tuple, Union

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.robot_warehouse.constants import _AGENTS, _SHELVES
from jumanji.environments.routing.robot_warehouse.types import Action, Agent, Position
from jumanji.environments.routing.robot_warehouse.utils_shelf import (
    set_new_shelf_position_if_carrying,
)
from jumanji.tree_utils import tree_add_element, tree_slice


def update_agent(
    agents: Agent,
    agent_id: chex.Array,
    attr: str,
    value: Union[chex.Array, Position],
) -> Agent:
    """Update the attribute information of a specific agent.

    Args:
        agents: a pytree of either Agent type containing agent information.
        agent_id: unique ID identifying a specific agent.
        attr: the attribute to update, e.g. `direction`, or `is_requested`.
        value: the new value to which the attribute is to be set.

    Returns:
        the agent with the specified attribute updated to the given value.
    """
    params = {attr: value}
    agent = tree_slice(agents, agent_id)
    agent = agent._replace(**params)
    agents: Agent = tree_add_element(agents, agent_id, agent)
    return agents


def get_new_direction_after_turn(
    action: chex.Array, agent_direction: chex.Array
) -> chex.Array:
    """Get the correct direction the agent should face given
    the turn action it took. E.g. if the agent is facing LEFT
    and turns RIGHT it should now be facing UP, etc.

    Args:
        action: the agent's action.
        agent_direction: the agent's current direction.

    Returns:
        the direction the agent should be facing given the action it took.
    """
    change_in_direction = jnp.array([0, 0, -1, 1, 0])[action]
    return (agent_direction + change_in_direction) % 4


def get_new_position_after_forward(
    grid: chex.Array, agent_position: chex.Array, agent_direction: chex.Array
) -> Position:
    """Get the correct position the agent will be in after moving forward
    in its current direction. E.g. if the agent is facing LEFT and turns
    RIGHT it should stay in the same position. If instead it moves FORWARD
    it should move left by one cell.

    Args:
        grid: the warehouse floor grid array.
        agent_position: the agent's current position.
        agent_direction: the agent's current direction.

    Returns:
        the position the agent should be in given the action it took.
    """
    _, grid_width, grid_height = grid.shape
    x, y = agent_position.x, agent_position.y
    move_up = lambda x, y: Position(jnp.max(jnp.array([0, x - 1])), y)
    move_right = lambda x, y: Position(x, jnp.min(jnp.array([grid_height - 1, y + 1])))
    move_down = lambda x, y: Position(jnp.min(jnp.array([grid_width - 1, x + 1])), y)
    move_left = lambda x, y: Position(x, jnp.max(jnp.array([0, y - 1])))
    new_position: Position = jax.lax.switch(
        agent_direction, [move_up, move_right, move_down, move_left], x, y
    )
    return new_position


def get_agent_view(
    grid: chex.Array, agent: chex.Array, sensor_range: chex.Array
) -> Tuple[chex.Array, chex.Array]:
    """Get an agent's view of other agents and shelves within its
    sensor range.

    Below is an example of the agent's view of other agents from
    the perspective of agent 1 with a sensor range of 1:

                            0, 0, 0
                            0, 1, 2
                            0, 0, 0

    It sees agent 2 to its right. Separately, the view of shelves
    is shown below:

                            0, 0, 0
                            0, 3, 4
                            0, 7, 8

    Agent 1 is on top of shelf 3 and has 4, 7 and 8 around it in
    the bottom right corner of its view. Before returning these
    views they are flattened into a 1-d arrays, i.e.

    View of agents: [0, 0, 0, 0, 1, 2, 0, 0, 0]
    View of shelves: [0, 0, 0, 0, 3, 4, 0, 7, 8]


    Args:
        grid: the warehouse floor grid array.
        agent: the agent for which the view of their receptive field
            is to be calculated.
        sensor_range: the range of the agent's sensors.

    Returns:
        a view of the agents receptive field separated into two arrays:
        one for other agents and one for shelves.
    """
    receptive_field = sensor_range * 2 + 1
    padded_agents_layer = jnp.pad(grid[_AGENTS], sensor_range, mode="constant")
    padded_shelves_layer = jnp.pad(grid[_SHELVES], sensor_range, mode="constant")
    agent_view_of_agents = jax.lax.dynamic_slice(
        padded_agents_layer,
        (agent.position.x, agent.position.y),
        (receptive_field, receptive_field),
    ).reshape(-1)
    agent_view_of_shelves = jax.lax.dynamic_slice(
        padded_shelves_layer,
        (agent.position.x, agent.position.y),
        (receptive_field, receptive_field),
    ).reshape(-1)
    return agent_view_of_agents, agent_view_of_shelves


def set_agent_carrying_if_at_shelf_position(
    grid: chex.Array, agents: chex.Array, agent_id: int, is_highway: chex.Array
) -> chex.Array:
    """Set the agent as carrying a shelf if it is at a shelf position.

    Args:
        grid: the warehouse floor grid array.
        agents: a pytree of either Agent type containing agent information.
        agent_id: unique ID identifying a specific agent.
        is_highway: binary value indicating highway position.

    Returns:
        updated agents pytree.
    """
    agent = tree_slice(agents, agent_id)
    shelf_id = grid[_SHELVES, agent.position.x, agent.position.y]

    return jax.lax.cond(
        shelf_id > 0,
        lambda: update_agent(agents, agent_id, "is_carrying", 1),
        lambda: agents,
    )


def offload_shelf_if_position_is_open(
    grid: chex.Array, agents: chex.Array, agent_id: int, is_highway: chex.Array
) -> chex.Array:
    """Set the agent as not carrying a shelf if it is at a shelf position.

    Args:
        grid: the warehouse floor grid array.
        agents: a pytree of either Agent type containing agent information.
        agent_id: unique ID identifying a specific agent.
        is_highway: binary value indicating highway position.

    Returns:
        updated agents pytree.
    """
    return jax.lax.cond(
        jnp.logical_not(is_highway),
        lambda: update_agent(agents, agent_id, "is_carrying", 0),
        lambda: agents,
    )


def set_carrying_shelf_if_load_toggled_and_not_carrying(
    grid: chex.Array,
    agents: chex.Array,
    action: int,
    agent_id: int,
    is_highway: chex.Array,
) -> chex.Array:
    """Set the agent as carrying a shelf if the load toggle action is
    performed and the agent is not carrying a shelf.

    Args:
        grid: the warehouse floor grid array.
        agents: a pytree of either Agent type containing agent information.
        action: the agent's action.
        agent_id: unique ID identifying a specific agent.
        is_highway: binary value indicating highway position.

    Returns:
        updated agents pytree.
    """
    agent = tree_slice(agents, agent_id)

    agents = jax.lax.cond(
        (action == Action.TOGGLE_LOAD.value) & ~agent.is_carrying,
        set_agent_carrying_if_at_shelf_position,
        offload_shelf_if_position_is_open,
        grid,
        agents,
        agent_id,
        is_highway,
    )
    return agents


def rotate_agent(
    grid: chex.Array,
    agents: chex.Array,
    action: int,
    agent_id: int,
    is_highway: chex.Array,
) -> chex.Array:
    """Rotate the agent in the direction of the action.

    Args:
        grid: the warehouse floor grid array.
        agents: a pytree of either Agent type containing agent information.
        action: the agent's action.
        agent_id: unique ID identifying a specific agent.
        is_highway: binary value indicating highway position.

    Returns:
        updated agents pytree.
    """
    agent = tree_slice(agents, agent_id)
    new_direction = get_new_direction_after_turn(action, agent.direction)
    return update_agent(agents, agent_id, "direction", new_direction)


def set_new_position_after_forward(
    grid: chex.Array,
    agents: chex.Array,
    shelves: chex.Array,
    action: int,
    agent_id: int,
    is_highway: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Set the new position of the agent after a forward action.

    Args:
        grid: the warehouse floor grid array.
        agents: a pytree of either Agent type containing agent information.
        shelves: a pytree of Shelf type containing shelf information.
        action: the agent's action.
        agent_id: unique ID identifying a specific agent.
        is_highway: binary value indicating highway position.

    Returns:
        updated grid array, agents and shelves pytrees.
    """
    # update agent position
    agent = tree_slice(agents, agent_id)
    current_position = agent.position
    new_position = get_new_position_after_forward(grid, agent.position, agent.direction)
    agents = update_agent(agents, agent_id, "position", new_position)

    # update agent grid placement
    grid = grid.at[_AGENTS, current_position.x, current_position.y].set(0)
    grid = grid.at[_AGENTS, new_position.x, new_position.y].set(agent_id + 1)

    grid, shelves = jax.lax.cond(
        agent.is_carrying,
        set_new_shelf_position_if_carrying,
        lambda g, s, p, np: (g, s),
        grid,
        shelves,
        current_position,
        new_position,
    )
    return grid, agents, shelves


def set_new_direction_after_turn(
    grid: chex.Array,
    agents: chex.Array,
    shelves: chex.Array,
    action: int,
    agent_id: int,
    is_highway: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Set the new direction of the agent after a turning action.

    Args:
        grid: the warehouse floor grid array.
        agents: a pytree of either Agent type containing agent information.
        shelves: a pytree of Shelf type containing shelf information.
        action: the agent's action.
        agent_id: unique ID identifying a specific agent.
        is_highway: binary value indicating highway position.

    Returns:
        updated grid array, agents and shelves pytrees.
    """
    agents = jax.lax.cond(
        jnp.isin(action, jnp.array([Action.LEFT.value, Action.RIGHT.value])),
        rotate_agent,
        set_carrying_shelf_if_load_toggled_and_not_carrying,
        grid,
        agents,
        action,
        agent_id,
        is_highway,
    )
    return grid, agents, shelves
