import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.lbf.types import Agent
from jumanji.environments.routing.lbf.utils import check_collision


def test_check_collision():
    orig_agents = jax.vmap(Agent)(
        id=jnp.array([0, 1, 2, 3]),
        position=jnp.array([[0, 1], [2, 2], [1, 0], [1, 1]]),
        level=jnp.array([0, 1, 2, 3]),
        fov=jnp.array([1, 2, 3, 4]),
    )
    moved_agents = jax.vmap(Agent)(
        id=jnp.array([0, 1, 2, 3]),
        # collision on agent 0 and 3
        position=jnp.array([[0, 0], [2, 2], [2, 0], [0, 0]]),
        level=jnp.array([0, 1, 2, 3]),
        fov=jnp.array([1, 2, 3, 4]),
    )

    expected_agents = jax.vmap(Agent)(
        id=jnp.array([0, 1, 2, 3]),
        # take orig agent for agent 0 and 3
        position=jnp.array([[0, 1], [2, 2], [2, 0], [1, 1]]),
        level=jnp.array([0, 1, 2, 3]),
        fov=jnp.array([1, 2, 3, 4]),
    )

    new_agents = check_collision(moved_agents, orig_agents)
    chex.assert_trees_all_equal(new_agents, expected_agents)
