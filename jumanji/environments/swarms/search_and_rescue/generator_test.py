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
import jax.numpy as jnp
import pytest

from jumanji.environments.swarms.common.types import AgentParams
from jumanji.environments.swarms.search_and_rescue.generator import Generator, RandomGenerator
from jumanji.environments.swarms.search_and_rescue.types import State


@pytest.mark.parametrize("env_size", [1.0, 0.5])
def test_random_generator(key: chex.PRNGKey, env_size: float) -> None:
    params = AgentParams(
        max_rotate=0.5,
        max_accelerate=0.01,
        min_speed=0.01,
        max_speed=0.05,
        view_angle=0.5,
    )
    generator = RandomGenerator(num_searchers=100, num_targets=101, env_size=env_size)

    assert isinstance(generator, Generator)

    state = generator(key, params)

    assert isinstance(state, State)
    assert state.searchers.pos.shape == (generator.num_searchers, 2)
    assert jnp.all(0.0 <= state.searchers.pos) and jnp.all(state.searchers.pos <= env_size)
    assert state.targets.pos.shape == (generator.num_targets, 2)
    assert jnp.all(0.0 <= state.targets.pos) and jnp.all(state.targets.pos <= env_size)
    assert not jnp.any(state.targets.found)
    assert state.step == 0
