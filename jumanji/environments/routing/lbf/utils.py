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

from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.lbf.constants import LOAD, MOVES
from jumanji.environments.routing.lbf.types import Agent, Entity, Food


def are_entities_adjacent(entity_a: Entity, entity_b: Entity) -> chex.Array:
    """
    Check if two entities are adjacent in the grid.

    Args:
        entity_a (Entity): The first entity.
        entity_b (Entity): The second entity.

    Returns:
        chex.Array: True if entities are adjacent, False otherwise.
    """
    distance = jnp.abs(entity_a.position - entity_b.position)
    return jnp.where(jnp.sum(distance) == 1, True, False)


def flag_duplicates(a: chex.Array) -> chex.Array:
    """Return a boolean array indicating which elements of `a` are duplicates.

    Example:
        a = jnp.array([1, 2, 3, 2, 1, 5])
        flag_duplicates(a)  # jnp.array([True, False, True, False, True, True])
    """
    # https://stackoverflow.com/a/11528078/5768407
    _, indices, counts = jnp.unique(
        a, return_inverse=True, return_counts=True, size=len(a), axis=0
    )
    return ~(counts[indices] == 1)


def simulate_agent_movement(
    agent: Agent, action: chex.Array, food_items: Food, agents: Agent, grid_size: int
) -> Agent:
    """
    Move the agent based on the specified action.

    Args:
        agent (Agent): The agent to move.
        action (chex.Array): The action to take.
        food_items (Food): All food items in the grid.
        agents (Agent): All agents in the grid.
        grid_size (int): The size of the grid.

    Returns:
        Agent: The agent with its updated position.
    """

    # Calculate the new position based on the chosen action
    new_position = agent.position + MOVES[action]

    # Check if the new position is out of bounds
    out_of_bounds = jnp.any((new_position < 0) | (new_position >= grid_size))

    # Check if the new position is occupied by food or another agent
    agent_at_position = jnp.any(
        jnp.all(new_position == agents.position, axis=1) & (agent.id != agents.id)
    )
    food_at_position = jnp.any(
        jnp.all(new_position == food_items.position, axis=1) & ~food_items.eaten
    )
    entity_at_position = jnp.any(agent_at_position | food_at_position)

    # Move the agent to the new position if it's a valid position,
    # otherwise keep the current position
    new_agent_position = jnp.where(
        out_of_bounds | entity_at_position, agent.position, new_position
    )

    # Return the agent with the updated position
    return Agent(id=agent.id, position=new_agent_position, level=agent.level)


def update_agent_positions(
    agents: Agent, actions: chex.Array, food_items: Food, grid_size: int
) -> Any:
    """
    Update agent positions based on actions, resolve collisions, and set loading status.

    Args:
        agents (Agent): The current state of agents.
        actions (chex.Array): Actions taken by agents.
        food_items (Food): All food items in the grid.
        grid_size (int): The size of the grid.

    Returns:
        Agent: Agents with updated positions and loading status.
    """
    # Move the agent to a valid position
    moved_agents = jax.vmap(simulate_agent_movement, (0, 0, None, None, None))(
        agents,
        actions,
        food_items,
        agents,
        grid_size,
    )

    # Fix collisions
    moved_agents = fix_collisions(moved_agents, agents)

    # set agent's loading status
    moved_agents = jax.vmap(
        lambda agent, action: agent.replace(loading=(action == LOAD))
    )(moved_agents, actions)

    return moved_agents


def fix_collisions(moved_agents: Agent, original_agents: Agent) -> Agent:
    """
    Fix collisions in the moved agents by resolving conflicts with the original agents.
    If a number 'N' of agents end up in the same position after the move, the initial
    position of the agents is retained.

    Args:
        moved_agents (Agent): Agents with potentially updated positions.
        original_agents (Agent): Original agents with their initial positions.

    Returns:
        Agent: Agents with collisions resolved.
    """
    # Detect duplicate positions
    duplicates = flag_duplicates(moved_agents.position)
    # If there are duplicates, use the original agent position.
    new_positions = jnp.where(
        duplicates,
        original_agents.position,
        moved_agents.position,
    )

    # Recreate agents with new positions
    agents: Agent = jax.vmap(Agent)(
        id=original_agents.id,
        position=new_positions,
        level=original_agents.level,
        loading=original_agents.loading,
    )
    return agents


def eat_food(agents: Agent, food: Food) -> Tuple[Food, chex.Array, chex.Array]:
    """Try to eat the provided food if possible.

    Args:
        agents(Agent): All agents in the grid.
        food(Food): The food to attempt to eat.

    Returns:
        new_food (Food): Updated state of the food, indicating whether it was eaten.
        food_eaten_this_step (chex.Array): Whether or not the food was eaten at this step.
        agents_loading_levels (chex.Array): Adjacent agents' levels loading around the food.
    """

    def get_adjacent_levels(agent: Agent, food: Food) -> chex.Array:
        """Return the level of the agent if it is adjacent to the food, else 0."""
        return jax.lax.select(
            are_entities_adjacent(agent, food) & agent.loading & ~food.eaten,
            agent.level,
            0,
        )

    # Get the level of all adjacent agents that are trying to load the food
    adj_loading_agents_levels = jax.vmap(get_adjacent_levels, (0, None))(agents, food)

    # If the food has already been eaten or is not loaded, the sum will be equal to 0
    food_eaten_this_step = jnp.sum(adj_loading_agents_levels) >= food.level

    # Set food to eaten if it was eaten.
    new_food = food.replace(eaten=food_eaten_this_step | food.eaten)  # type: ignore

    return new_food, food_eaten_this_step, adj_loading_agents_levels


def place_agent_on_grid(agent: Agent, grid: chex.Array) -> chex.Array:
    """Return the grid with the agent placed on it."""
    x, y = agent.position
    return grid.at[x, y].set(agent.level)


def place_food_on_grid(food: Food, grid: chex.Array) -> chex.Array:
    """Return the grid with the food placed on it."""
    x, y = food.position
    return grid.at[x, y].set(food.level * ~food.eaten)  # 0 if eaten else level


def slice_around(pos: chex.Array, fov: int) -> Tuple[chex.Array, chex.Array]:
    """Return the start and length of a slice that when used to index a grid will
    return a 2*fov+1 x 2*fov+1 sub-grid centered around pos.

    Returns are meant to be used with a `jax.lax.dynamic_slice`
    """
    # Because we pad the grid by fov we need to shift the pos to the position
    # it will be in the padded grid.
    shifted_pos = pos + fov

    start_x = shifted_pos[0] - fov
    start_y = shifted_pos[1] - fov
    return start_x, start_y


def calculate_num_observation_features(num_food: int, num_agents: int) -> chex.Array:
    """Calculate the number of features in an agent view"""
    obs_features = 3 * (num_food + num_agents)
    return jnp.array(obs_features, jnp.int32)
