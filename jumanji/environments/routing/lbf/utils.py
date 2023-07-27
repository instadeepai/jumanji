from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.lbf.types import Agent, Entity, Food


def place_agent_on_grid(agent: Agent, grid: chex.Array) -> chex.Array:
    return grid.at[agent.position].set(agent.id)


def move(
    agent: Agent, action: chex.Array, foods: Food, max_row: int, max_col: int
) -> Agent:
    movements = jnp.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]])
    bounds = jnp.array([max_row, max_col])

    # add action to agent position
    new_position = agent.position + movements[action]

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


def eat(agents: Agent, food: Food) -> Tuple[bool, Food]:
    """Return whether any agents ate any food, and the new food."""
    # get the level of all adjacent agents, if an agent is not adjacent, it's level is 0
    adjacent_levels = jax.vmap(
        lambda agent, food: jax.lax.select(
            is_adj(agent, food),
            agent.level,
            0,
        )
    )(agents, food)

    # sum the levels of all adjacent agents that are loading
    adjacent_level = jnp.sum(jnp.where(agents.loading, adjacent_levels, 0))

    # todo: check if greater than equal to
    food_eaten = adjacent_level >= food.level
    return food_eaten, food.replace(eaten=food_eaten)


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
