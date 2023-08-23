from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.lbf.constants import MOVES
from jumanji.environments.routing.lbf.types import Agent, Entity, Food


def place_agent_on_grid(agent: Agent, grid: chex.Array) -> chex.Array:
    """Return the grid with the agent placed on it."""
    x, y = agent.position
    return grid.at[x, y].set(agent.level)
    # jax.debug.print("ag pos: {a}, {b}", a=x, b=y)
    # print(f"x, y = {x.shape}, {y.shape}")

    # I don't like this, but can't find a better way to index a 2D array with another 2D array
    # inds = jax.vmap(tuple)(agent.position)
    # return grid.at[inds].set(agent.level)


def place_food_on_grid(food: Food, grid: chex.Array) -> chex.Array:
    """Return the grid with the food placed on it."""
    x, y = food.position
    return jax.lax.select(food.eaten, grid, grid.at[x, y].set(food.level))


def move(agent: Agent, action: chex.Array, foods: Food, grid_size: int) -> Agent:
    """Return the new agent position after taking `action`.

    Agent cannot move to a position that is occupied by food or out of bounds.
    """
    # add action to agent position
    new_position = agent.position + MOVES[action]

    # if position is not in food positions and not out of bounds, move agent
    out_of_bounds = (new_position < 0) | (new_position >= grid_size)
    invalid_position = jnp.any(jnp.all(new_position == foods.position, axis=1))

    return agent.replace(
        position=jnp.where(
            out_of_bounds | invalid_position, agent.position, new_position
        )
    )


def is_adj(a: Entity, b: Entity) -> bool:
    """Return whether `a` and `b` are adjacent."""
    # todo: would it be quicker to just:
    # (abs(a.position[0] - b.position[0]) == 1 and a.position[1] == b.position[1]) or (a.position[0] == b.position[0] and abs(a.position[1] - b.position[1]) == 1)
    return jnp.linalg.norm(a.position - b.position, axis=-1) == 1


def eat(agents: Agent, food: Food) -> Tuple[Food, chex.Array, chex.Array]:
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
    return new_food, food_eaten, adjacent_loading_levels


def flag_duplicates(a: chex.Array):
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


def fix_collisions(moved_agents: Agent, orig_agents: Agent) -> Agent:
    """Return agents with collisions fixed, if two agents are in the same position, use the original agent position."""
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


def slice_around(pos: chex.Array, fov: int) -> Tuple[chex.Array, chex.Array]:
    """Return the start and length of a slice that when used to index a grid will
    return a 2*fov+1 x 2*fov+1 sub-grid centered around pos.

    Returns are meant to be used with a `jax.lax.dynamic_slice`
    """
    # because we pad the grid by fov we need to shift the pos to the position it will be in the padded grid
    shifted_pos = pos + fov

    start_x = shifted_pos[0] - fov
    start_y = shifted_pos[1] - fov
    return start_x, start_y
