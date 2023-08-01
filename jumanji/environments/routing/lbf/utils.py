from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.lbf.constants import MOVES
from jumanji.environments.routing.lbf.types import Agent, Entity, Food


def place_agent_on_grid(agent: Agent, grid: chex.Array) -> chex.Array:
    # todo: this places the agent on the grid, how does lbf display the agent's level in the obs without obstructing the food level?
    return grid.at[agent.position].set(agent.level)


def place_food_on_grid(food: Food, grid: chex.Array) -> chex.Array:
    return jax.lax.select(food.eaten, grid, grid.at[food.position].set(food.level))


def move(agent: Agent, action: chex.Array, foods: Food, grid_size: int) -> Agent:
    bounds = jnp.array([grid_size, grid_size])

    # add action to agent position
    new_position = agent.position + MOVES[action]

    # if position is not in food positions and not out of bounds, move agent
    out_of_bounds = jnp.any(new_position < 0) | jnp.any(new_position >= bounds)
    invalid_position = jnp.any(agent.position == foods.position, axis=-1)

    return agent.replace(
        position=jnp.where(
            out_of_bounds | invalid_position, agent.position, new_position
        )
    )


def is_adj(a: Entity, b: Entity) -> bool:
    """Return whether `a` and `b` are adjacent."""
    return jnp.linalg.norm(a.position - b.position, axis=-1) == 1


def eat(agents: Agent, food: Food) -> Tuple[Food, bool, chex.Array]:
    """Return the new food, whether any agents ate any food and the agents that were loading around the food."""

    def get_adj_level(agent: Agent, food: Food) -> chex.Array:
        return jax.lax.select(
            is_adj(agent, food),
            agent.level,
            0,
        )

    # get the level of all adjacent agents, if an agent is not adjacent, it's level is 0
    adjacent_levels = jax.vmap(get_adj_level, (0, None))(agents, food)

    # sum the levels of all adjacent agents that are loading
    adjacent_loading_levels = jnp.where(agents.loading, adjacent_levels, 0)
    adjacent_level = jnp.sum(adjacent_loading_levels)

    # todo: check if greater than equal to or just greater than
    food_eaten = (adjacent_level >= food.level) & (~food.eaten)
    # set food to eaten if it was eaten and if it was already eaten leave it as eaten
    new_food = food.replace(eaten=food_eaten | food.eaten)
    return new_food, food_eaten, adjacent_loading_levels != 0


def flag_duplicates(a: chex.Array):
    """Return a boolean array indicating which elements of `a` are duplicates.

    Example:
        a = jnp.array([1, 2, 3, 2, 1, 5])
        flag_duplicates(a)  # jnp.array([True, False, True, False, True, True])
    """
    _, indices, counts = jnp.unique(a, return_inverse=True, return_counts=True, axis=0)
    return ~(counts[indices] == 1)


def fix_collisions(moved_agents: Agent, orig_agents: Agent) -> Agent:
    duplicates = flag_duplicates(moved_agents.position)
    # need to broadcast this so the where works
    duplicates = jnp.broadcast_to(duplicates[:, None], orig_agents.position.shape)

    # if there are duplicates, use the original agent position
    new_positions = jnp.where(
        duplicates,
        orig_agents.position,
        moved_agents.position,
    )

    # recreate agents with new positions
    return jax.vmap(Agent)(
        id=orig_agents.id,
        position=new_positions,
        level=orig_agents.level,
        loading=orig_agents.loading,
    )
