import jax.numpy as jnp
import pytest

from jumanji.environments.routing.lbf.env import LevelBasedForaging
from jumanji.environments.routing.lbf.generator import UniformRandomGenerator
from jumanji.environments.routing.lbf.types import Agent, Food
from jumanji.tree_utils import tree_transpose

# create food and agents for grid that looks like:
# AGENT | AGENT | EMPTY
# AGENT | FOOD  | AGENT
# FOOD  | EMPTY | EMPTY


@pytest.fixture
def agent0() -> Agent:
    return Agent(
        id=jnp.asarray(0),
        position=jnp.array([0, 0]),
        level=jnp.asarray(1),
        loading=jnp.asarray(False),
    )


@pytest.fixture
def agent1() -> Agent:
    return Agent(
        id=jnp.asarray(1),
        position=jnp.array([0, 1]),
        level=jnp.asarray(2),
        loading=jnp.asarray(False),
    )


@pytest.fixture
def agent2() -> Agent:
    return Agent(
        id=jnp.asarray(2),
        position=jnp.array([1, 0]),
        level=jnp.asarray(2),
        loading=jnp.asarray(False),
    )


@pytest.fixture
def agent3() -> Agent:
    return Agent(
        id=jnp.asarray(3),
        position=jnp.array([1, 2]),
        level=jnp.asarray(1),
        loading=jnp.asarray(False),
    )


@pytest.fixture
def food0() -> Food:
    return Food(
        id=jnp.asarray(0),
        position=jnp.array([1, 1]),
        level=jnp.asarray(4),
    )


@pytest.fixture
def food1() -> Food:
    return Food(
        id=jnp.asarray(1),
        position=jnp.array([2, 0]),
        level=jnp.asarray(3),
    )


@pytest.fixture
def agents(agent0: Agent, agent1: Agent, agent2: Agent, agent3: Agent) -> Agent:
    return tree_transpose([agent0, agent1, agent2, agent3])


@pytest.fixture
def foods(food0: Food, food1: Food) -> Food:
    return tree_transpose([food0, food1])


@pytest.fixture
def level_based_foraging_env():
    generator = UniformRandomGenerator(
        grid_size=3, num_agents=4, num_food=2, max_agent_level=2, max_food_level=4
    )
    return LevelBasedForaging(generator=generator, fov=1, max_steps=500)
