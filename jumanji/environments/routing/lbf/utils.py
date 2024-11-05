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
from jumanji.environments.routing.lbf.types import Agent, Entity, Food, State


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
    return jnp.sum(distance) == 1


def flag_duplicates(a: chex.Array) -> chex.Array:
    """Return a boolean array indicating which elements of `a` are duplicates.

    Example:
        a = jnp.array([1, 2, 3, 2, 1, 5])
        flag_duplicates(a)  # jnp.array([True, False, True, False, True, True])
    """
    # https://stackoverflow.com/a/11528078/5768407
    _, indices, counts = jnp.unique(a, return_inverse=True, return_counts=True, size=len(a), axis=0)
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
    new_agent_position = jnp.where(out_of_bounds | entity_at_position, agent.position, new_position)

    # Return the agent with the updated position
    return agent.replace(position=new_agent_position)  # type: ignore


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
    moved_agents = jax.vmap(lambda agent, action: agent.replace(loading=(action == LOAD)))(
        moved_agents, actions
    )

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
    duplicates = duplicates.reshape((duplicates.shape[0], -1))

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


def compute_action_mask(agent: Agent, state: State, grid_size: int) -> chex.Array:
    """
    Calculate the action mask for a given agent based on the current state.

    Args:
        agent (Agent): The agent for which to calculate the action mask.
        state (State): The current state of the environment.

    Returns:
        chex.Array: A boolean array representing the action mask for the given agent,
            where `True` indicates a valid action, and `False` indicates an invalid action.
    """
    next_positions = agent.position + MOVES

    def check_pos_fn(next_pos: Any, entities: Entity, condition: bool) -> Any:
        return jnp.any(jnp.all(next_pos == entities.position, axis=-1) & condition)

    # Check if any agent is in a next position.
    agent_occupied = jax.vmap(check_pos_fn, (0, None, None))(
        next_positions, state.agents, (state.agents.id != agent.id)
    )

    # Check if any food is in a next position (Food must be uneaten)
    food_occupied = jax.vmap(check_pos_fn, (0, None, None))(
        next_positions, state.food_items, ~state.food_items.eaten
    )
    # Check if the next position is out of bounds
    out_of_bounds = jnp.any((next_positions < 0) | (next_positions >= grid_size), axis=-1)

    action_mask = ~(food_occupied | agent_occupied | out_of_bounds)

    # Check if the agent can load food (if placed in the neighborhood)
    num_adj_food = (
        jax.vmap(are_entities_adjacent, (0, None))(state.food_items, agent)
        & ~state.food_items.eaten
    )
    is_food_adj = jnp.sum(num_adj_food) > 0

    action_mask = jnp.where(is_food_adj, action_mask, action_mask.at[-1].set(False))

    return action_mask
