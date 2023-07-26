from typing import List, Tuple

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.lbf.types import Agent, Food


def place_agent_on_grid(agent: Agent, grid: chex.Array) -> chex.Array:
    pass


def move(agent: Agent, action: chex.Array) -> Tuple[Agent, chex.Array]:
    pass


def eat(agents: Agent, food: Food, grid: chex.Array) -> Tuple[Food, chex.Array]:
    pass


def join_grids(grid1: chex.Array, grid2: chex.Array) -> chex.Array:
    pass


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
        fov=orig_agents.fov,
    )
