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
import pytest

from jumanji.environments.routing.robot_warehouse.generator import RandomGenerator


@pytest.fixture
def random_generator() -> RandomGenerator:
    """Creates a generator with 2 agents."""
    return RandomGenerator(
        shelf_rows=1,
        shelf_columns=3,
        column_height=2,
        num_agents=2,
        sensor_range=1,
        request_queue_size=4,
    )


def test_random_generator__call(random_generator: RandomGenerator) -> None:
    """Test that generator generates valid boards."""
    key = jax.random.PRNGKey(42)
    state = random_generator(key)
    grid_size = (2, 5, 10)
    assert state.grid.shape == grid_size
    assert state.agents.direction.shape[0] == 2


def test_random_generator__no_retrace(
    random_generator: RandomGenerator,
) -> None:
    """Checks that generator only traces the function once and works when jitted."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 2)
    jitted_generator = jax.jit(chex.assert_max_traces((random_generator.__call__), n=1))

    for key in keys:
        jitted_generator(key)
