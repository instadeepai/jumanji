import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.lbf.types import Agent
from jumanji.environments.routing.lbf.utils import fix_collisions, is_adj


def test_fix_collisions():
    orig_agents = jax.vmap(Agent)(
        id=jnp.array([0, 1, 2, 3]),
        position=jnp.array([[0, 1], [2, 2], [1, 0], [1, 1]]),
        level=jnp.array([0, 1, 2, 3]),
    )
    moved_agents = jax.vmap(Agent)(
        id=jnp.array([0, 1, 2, 3]),
        # collision on agent 0 and 3
        position=jnp.array([[0, 0], [2, 2], [2, 0], [0, 0]]),
        level=jnp.array([0, 1, 2, 3]),
    )

    expected_agents = jax.vmap(Agent)(
        id=jnp.array([0, 1, 2, 3]),
        # take orig agent for agent 0 and 3
        position=jnp.array([[0, 1], [2, 2], [2, 0], [1, 1]]),
        level=jnp.array([0, 1, 2, 3]),
    )

    new_agents = fix_collisions(moved_agents, orig_agents)
    chex.assert_trees_all_equal(new_agents, expected_agents)


def test_is_adj():
    a1 = Agent(id=0, position=jnp.array([0, 0]), level=0)
    a2 = Agent(id=1, position=jnp.array([0, 1]), level=0)
    a3 = Agent(id=2, position=jnp.array([1, 1]), level=0)

    many_agents = jax.vmap(Agent)(
        id=jnp.array([0, 1, 2]),
        position=jnp.array([[0, 0], [0, 1], [1, 1]]),
        level=jnp.array([0, 0, 0]),
    )

    assert is_adj(a1, a2)
    assert is_adj(a2, a3)
    assert not is_adj(a1, a3)

    assert jnp.all(
        jax.vmap(is_adj, (0, None))(many_agents, a1) == jnp.array([False, True, False])
    )
