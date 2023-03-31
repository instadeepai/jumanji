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

from jumanji.environments.routing.cleaner.constants import CLEAN, DIRTY, WALL
from jumanji.environments.routing.cleaner.generator import RandomGenerator


class TestRandomGenerator:
    WIDTH = 5
    HEIGHT = 7
    NUM_AGENTS = 3

    @pytest.fixture
    def key(self) -> chex.PRNGKey:
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def instance_generator(self) -> RandomGenerator:
        return RandomGenerator(self.WIDTH, self.HEIGHT, self.NUM_AGENTS)

    def test_random_instance_generator_values(
        self, key: chex.PRNGKey, instance_generator: RandomGenerator
    ) -> None:
        state = instance_generator(key)

        assert jnp.all(state.agents_locations == jnp.zeros((self.NUM_AGENTS, 2)))
        assert jnp.sum(jnp.logical_and(state.grid != WALL, state.grid != DIRTY)) == 1
        assert state.grid[0, 0] == CLEAN  # Only the top-left tile is clean
        assert state.step_count == 0
