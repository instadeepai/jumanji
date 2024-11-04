# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import chex
import jax
import jax.numpy as jnp
import pytest

from jumanji.environments.routing.lbf.env import LevelBasedForaging
from jumanji.environments.routing.lbf.generator import RandomGenerator
from jumanji.environments.routing.lbf.types import Agent, Food, State
from jumanji.tree_utils import tree_transpose

# create food and agents for grid that looks like:
# "AGENT" | EMPTY   | EMPTY   | EMPTY  | EMPTY | EMPTY
# EMPTY   | "AGENT" | EMPTY   | EMPTY  | EMPTY | EMPTY
# EMPTY   | "FOOD"  | "AGENT" | "FOOD" | EMPTY | EMPTY
# EMPTY   | EMPTY   | EMPTY   | EMPTY  | EMPTY | EMPTY
# EMPTY   | EMPTY   | "FOOD"  | EMPTY  | EMPTY | EMPTY
# EMPTY   | EMPTY   | EMPTY   | EMPTY  | EMPTY | EMPTY


@pytest.fixture
def key() -> chex.PRNGKey:
    return jax.random.PRNGKey(42)


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
        position=jnp.array([1, 1]),
        level=jnp.asarray(2),
        loading=jnp.asarray(False),
    )


@pytest.fixture
def agent2() -> Agent:
    return Agent(
        id=jnp.asarray(2),
        position=jnp.array([2, 2]),
        level=jnp.asarray(4),
        loading=jnp.asarray(False),
    )


@pytest.fixture
def food0() -> Food:
    return Food(
        id=jnp.asarray(0),
        position=jnp.array([2, 1]),
        level=jnp.asarray(4),
        eaten=jnp.asarray(False),
    )


@pytest.fixture
def food1() -> Food:
    return Food(
        id=jnp.asarray(1),
        position=jnp.array([2, 3]),
        level=jnp.asarray(4),
        eaten=jnp.asarray(False),
    )


@pytest.fixture
def food2() -> Food:
    return Food(
        id=jnp.asarray(1),
        position=jnp.array([4, 2]),
        level=jnp.asarray(3),
        eaten=jnp.asarray(False),
    )


@pytest.fixture
def agents(agent0: Agent, agent1: Agent, agent2: Agent) -> Agent:
    return tree_transpose([agent0, agent1, agent2])


@pytest.fixture
def food_items(food0: Food, food1: Food, food2: Food) -> Food:
    return tree_transpose([food0, food1, food2])


@pytest.fixture
def state(agents: Agent, food_items: Food, key: chex.PRNGKey) -> State:
    return State(agents=agents, food_items=food_items, step_count=0, key=key)


@pytest.fixture
def agent_grid() -> chex.Array:
    """Returns the agents' levels in their postion on the grid."""
    return jnp.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0],
            [0, 0, 4, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )


@pytest.fixture
def food_grid() -> chex.Array:
    """Returns the food items's levels in their postion on the grid."""
    return jnp.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 4, 0, 4, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )


@pytest.fixture
def random_generator() -> RandomGenerator:
    return RandomGenerator(
        grid_size=8,
        fov=2,
        num_agents=2,
        num_food=2,
        max_agent_level=2,
        force_coop=True,
    )


@pytest.fixture
def lbf_environment() -> LevelBasedForaging:
    generator = RandomGenerator(
        grid_size=8,
        fov=6,
        num_agents=3,
        num_food=3,
        max_agent_level=4,
        force_coop=True,
    )

    return LevelBasedForaging(generator=generator, time_limit=5)


@pytest.fixture
def lbf_env_2s() -> LevelBasedForaging:
    generator = RandomGenerator(
        grid_size=8,
        fov=2,
        num_agents=2,
        num_food=2,
        max_agent_level=2,
        force_coop=False,
    )

    return LevelBasedForaging(generator=generator, time_limit=5)


@pytest.fixture
def lbf_env_grid_obs() -> LevelBasedForaging:
    generator = RandomGenerator(
        grid_size=8,
        fov=6,
        num_agents=3,
        num_food=3,
        max_agent_level=4,
        force_coop=True,
    )

    return LevelBasedForaging(generator=generator, grid_observation=True)


@pytest.fixture
def lbf_with_penalty() -> LevelBasedForaging:
    return LevelBasedForaging(penalty=1.0)


@pytest.fixture
def lbf_with_no_norm_reward() -> LevelBasedForaging:
    return LevelBasedForaging(normalize_reward=False)
