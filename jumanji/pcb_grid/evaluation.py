import functools

import jax
import jax.numpy as jnp
from chex import Array

from jumanji.pcb_grid.constants import SOURCE
from jumanji.pcb_grid.env import PcbGridEnv


def is_episode_finished(env: PcbGridEnv, grid: Array) -> jnp.bool_:
    """Returns True if all agents are finished and the episode is completed."""
    dones = env.get_finished_agents(grid)
    return jnp.all(dones)


def proportion_connected(env: PcbGridEnv, grid: Array) -> float:
    """Calculates the proportion of wires that are connected."""
    connected_agents = jax.vmap(functools.partial(env.is_agent_connected, grid))(
        jnp.arange(env.num_agents)
    )
    proportion: float = jnp.mean(connected_agents).item()
    return proportion


def is_board_complete(env: PcbGridEnv, grid: Array) -> jnp.bool_:
    """Returns True if all agents in a state are connected.

    Args:
        grid : Any observation of the environment grid.

    Return:
        True if all agents are connected otherwise False.
    """
    return proportion_connected(env, grid) == 1


def wire_length(env: PcbGridEnv, grid: Array) -> int:
    """Calculates the length of all the wires on the grid."""
    total_wire_length: int = jnp.sum(
        jax.vmap(lambda i: jnp.count_nonzero(grid == SOURCE + 3 * i))(
            jnp.arange(env.num_agents)
        ),
        dtype=int,
    )
    return total_wire_length
