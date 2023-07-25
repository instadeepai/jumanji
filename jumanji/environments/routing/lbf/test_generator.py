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

    assert state.grid.shape == (grid_size, grid_size)
    assert jnp.sum((state.grid > 0) & (state.grid <= num_agents)) == num_agents
    assert jnp.sum(state.grid > num_agents) == num_food
