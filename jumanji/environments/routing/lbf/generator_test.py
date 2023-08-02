import jax
import jax.numpy as jnp

from jumanji.environments.routing.lbf.generator import UniformRandomGenerator


def test_generator():
    key = jax.random.PRNGKey(42)

    grid_size = 6
    num_agents = 7
    num_food = 8
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
