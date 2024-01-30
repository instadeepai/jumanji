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

from jumanji.environments.routing.pac_man.constants import DEFAULT_MAZE
from jumanji.environments.routing.pac_man.generator import AsciiGenerator


class TestAsciiGenerator:
    @pytest.fixture
    def key(self) -> chex.PRNGKey:
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def instance_generator(self) -> AsciiGenerator:
        return AsciiGenerator(DEFAULT_MAZE)

    def test_ascii_instance_generator_values(
        self,
        key: chex.PRNGKey,
        instance_generator: AsciiGenerator,
    ) -> None:
        state = instance_generator(key)

        assert state.step_count == 0
        assert state.grid.shape[0] == 31
        assert state.grid.shape[1] == 28
        assert state.pellets == 318
        assert state.frightened_state_time == 0
        assert jnp.array_equal(state.old_ghost_locations, state.ghost_locations)
        assert state.dead is False
