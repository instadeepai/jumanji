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

import jumanji.environments.routing.lbf.utils as utils
from jumanji.environments.routing.lbf.generator import RandomGenerator
from jumanji.environments.routing.lbf.types import State


def test_random_generator_call(random_generator: RandomGenerator, key: chex.PRNGKey) -> None:
    state = random_generator(key)
    assert random_generator.grid_size >= 5
    assert 2 <= random_generator.fov <= random_generator.grid_size
    assert isinstance(state, State)
    chex.assert_equal_shape([state.food_items.position[0], state.food_items.level])
    chex.assert_equal_shape([state.agents.position[0], state.agents.level])

    # Test that food levels are within a reasonable range if force_coop is True
    if random_generator.force_coop:
        max_food_level = jnp.sum(jnp.sort(state.agents.level)[:3])
        levels = jnp.full(shape=(random_generator.num_food,), fill_value=max_food_level)
        assert jnp.all(jnp.allclose(state.food_items.level, levels))

    # Check if no two food items are adjacent
    are_entities_adjacent = jax.vmap(utils.are_entities_adjacent, in_axes=(0, None))(
        state.food_items, state.food_items
    )
    assert not jnp.any(are_entities_adjacent)


def test_sample_food(random_generator: RandomGenerator, key: chex.PRNGKey) -> None:
    food_positions = random_generator.sample_food(key)

    # Check if positions are within the grid bounds and no food on the edge of the grid
    assert jnp.all((food_positions > 0) & (food_positions < random_generator.grid_size - 1))

    # Check if no food positions overlap
    assert not jnp.any(
        jnp.allclose(food_positions[:, None], food_positions)
        & ~jnp.eye(random_generator.num_agents, dtype=bool)
    )


def test_sample_agents(random_generator: RandomGenerator, key: chex.PRNGKey) -> None:
    mask = jnp.ones((random_generator.grid_size, random_generator.grid_size), dtype=bool)
    mask = mask.ravel()

    agent_positions = random_generator.sample_agents(key, mask)

    # Check if positions are within the grid bounds
    assert jnp.all((agent_positions >= 0) & (agent_positions < random_generator.grid_size))

    # Check if no agent positions overlap
    assert not jnp.any(
        jnp.allclose(agent_positions[:, None], agent_positions)
        & ~jnp.eye(random_generator.num_agents, dtype=bool)
    )


def test_sample_levels(random_generator: RandomGenerator, key: chex.PRNGKey) -> None:
    agent_levels = random_generator.sample_levels(
        random_generator.max_agent_level, (random_generator.num_agents,), key
    )

    # Check if levels are within the specified range
    assert jnp.all((agent_levels >= 1) & (agent_levels <= random_generator.max_agent_level))

    # Check if levels are generated randomly
    key2 = jax.random.PRNGKey(43)
    agent_levels2 = random_generator.sample_levels(
        random_generator.max_agent_level, (random_generator.num_agents,), key2
    )
    assert not jnp.all(jnp.allclose(agent_levels, agent_levels2))
