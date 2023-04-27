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

from jumanji.environments.routing.connector.generator import (
    ParallelRandomWalkGenerator,
    UniformRandomGenerator,
)
from jumanji.environments.routing.connector.utils import get_position, get_target


@pytest.fixture
def uniform_random_generator() -> UniformRandomGenerator:
    """Creates a generator with grid size of 5 and 3 agents."""
    return UniformRandomGenerator(grid_size=5, num_agents=3)


def test_uniform_random_generator__call(
    uniform_random_generator: UniformRandomGenerator, key: chex.PRNGKey
) -> None:
    """Test that generator generates valid boards."""
    state = uniform_random_generator(key)

    assert state.grid.shape == (5, 5)
    assert state.agents.position.shape == state.agents.target.shape == (3, 2)

    # Check grid has head and target for each agent
    # and the starts and ends point to the correct heads and targets, respectively
    agents_on_grid = state.grid[jax.vmap(tuple)(state.agents.position)]
    targets_on_grid = state.grid[jax.vmap(tuple)(state.agents.target)]
    assert (agents_on_grid == jnp.array([get_position(i) for i in range(3)])).all()
    assert (targets_on_grid == jnp.array([get_target(i) for i in range(3)])).all()


def test_uniform_random_generator__no_retrace(
    uniform_random_generator: UniformRandomGenerator, key: chex.PRNGKey
) -> None:
    """Checks that generator only traces the function once and works when jitted."""
    keys = jax.random.split(key, 2)
    jitted_generator = jax.jit(
        chex.assert_max_traces((uniform_random_generator.__call__), n=1)
    )

    for key in keys:
        jitted_generator(key)


class TestParallelRandomWalkGenerator:
    @pytest.fixture
    def parallel_random_walk_generator(self) -> ParallelRandomWalkGenerator:
        """Creates a generator with grid size of 5 and 3 agents."""
        return ParallelRandomWalkGenerator(grid_size=5, num_agents=3)

    def test_random_walk_generator__call(
        self,
        parallel_random_walk_generator: ParallelRandomWalkGenerator,
        key: chex.PRNGKey,
    ) -> None:
        """Tests that generator generates valid boards."""
        state = parallel_random_walk_generator(key)

        assert state.grid.shape == (5, 5)
        assert state.agents.position.shape == state.agents.target.shape == (3, 2)

        # Check grid has head and target for each agent
        # and the starts and ends point to the correct heads and targets, respectively
        agents_on_grid = state.grid[jax.vmap(tuple)(state.agents.position)]
        targets_on_grid = state.grid[jax.vmap(tuple)(state.agents.target)]
        assert (agents_on_grid == jnp.array([get_position(i) for i in range(3)])).all()
        # TODO: bug in placing targets on board
        assert (targets_on_grid == jnp.array([get_target(i) for i in range(3)])).all()

    def test_random_walk_generator__no_retrace(
        self,
        parallel_random_walk_generator: ParallelRandomWalkGenerator,
        key: chex.PRNGKey,
    ) -> None:
        """Checks that generator only traces the function once and works when jitted."""
        keys = jax.random.split(key, 2)
        jitted_generator = jax.jit(
            chex.assert_max_traces((parallel_random_walk_generator.__call__), n=1)
        )

        for key in keys:
            jitted_generator(key)
