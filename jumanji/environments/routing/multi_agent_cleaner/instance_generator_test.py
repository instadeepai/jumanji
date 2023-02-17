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

from jumanji.environments.routing.multi_agent_cleaner.constants import DIRTY, WALL
from jumanji.environments.routing.multi_agent_cleaner.instance_generator import (
    Maze,
    generate_random_instance,
)


class TestRandomInstanceGenerator:
    WIDTH = 5
    HEIGHT = 7

    @pytest.fixture
    def key(self) -> chex.PRNGKey:
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def instance(self, key: chex.PRNGKey) -> Maze:
        return generate_random_instance(self.WIDTH, self.HEIGHT, key)

    def test_random_instance_generator_values(self, instance: Maze) -> None:
        assert jnp.all(jnp.logical_or(instance == DIRTY, instance == WALL))
