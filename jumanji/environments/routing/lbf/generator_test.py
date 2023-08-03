import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.lbf.generator import UniformRandomGenerator


def is_adj(pos0: chex.Array, pos1: chex.Array) -> bool:
    return jnp.linalg.norm(pos0 - pos1) == 1


def test_generator():
    key = jax.random.PRNGKey(42)

    grid_size = 6
    num_agents = 7
    num_food = 6
    max_food_level = 9
    max_agent_level = 10

    gen = UniformRandomGenerator(
        grid_size, num_agents, num_food, max_agent_level, max_food_level
    )
    state = gen(key)

    # Test foods and agents placed within grid bounds.
    assert jnp.all(state.agents.position >= 0)
    assert jnp.all(state.agents.position < grid_size)
    assert jnp.all(state.foods.position >= 0)
    assert jnp.all(state.foods.position < grid_size)

    # test no foods are adjacent to eachother
    adjaciencies = jax.vmap(jax.vmap(is_adj, in_axes=(0, None)), in_axes=(None, 0))(
        state.foods.position, state.foods.position
    )
    assert jnp.all(~adjaciencies)

    # test no foods are on the edge of the grid
    assert jnp.all(state.foods.position != 0)
    assert jnp.all(state.foods.position != grid_size - 1)
