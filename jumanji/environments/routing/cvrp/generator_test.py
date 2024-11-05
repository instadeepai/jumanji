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

from jumanji.environments.routing.cvrp.conftest import DummyGenerator
from jumanji.environments.routing.cvrp.generator import UniformGenerator
from jumanji.environments.routing.cvrp.types import State
from jumanji.testing.pytrees import assert_trees_are_different, assert_trees_are_equal


class TestDummyGenerator:
    @pytest.fixture
    def dummy_generator(self) -> DummyGenerator:
        return DummyGenerator()

    def test_dummy_generator__properties(self, dummy_generator: DummyGenerator) -> None:
        """Validate that the dummy instance generator has the correct properties."""
        assert dummy_generator.num_nodes == 4
        assert dummy_generator.max_capacity == 6
        assert dummy_generator.max_demand == 3

    def test_dummy_generator__call(self, dummy_generator: DummyGenerator) -> None:
        """Validate that the dummy instance generator's call function behaves correctly,
        that it is jit-table and compiles only once, and that it returns the same state
        for different keys.
        """
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(dummy_generator.__call__, n=1))
        state1 = call_fn(jax.random.PRNGKey(1))
        state2 = call_fn(jax.random.PRNGKey(2))
        assert_trees_are_equal(state1, state2)


class TestUniformGenerator:
    @pytest.fixture
    def uniform_generator(self) -> UniformGenerator:
        return UniformGenerator(
            num_nodes=20,
            max_capacity=30,
            max_demand=10,
        )

    def test_uniform_generator__properties(self, uniform_generator: UniformGenerator) -> None:
        """Validate that the random instance generator has the correct properties."""
        assert uniform_generator.num_nodes == 20
        assert uniform_generator.max_capacity == 30
        assert uniform_generator.max_demand == 10

    def test_random_generator__call(self, uniform_generator: UniformGenerator) -> None:
        """Validate that the random instance generator's call function is jit-able and compiles
        only once. Also check that giving two different keys results in two different instances.
        """
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(uniform_generator.__call__, n=1))
        state1 = call_fn(key=jax.random.PRNGKey(1))
        assert isinstance(state1, State)

        state2 = call_fn(key=jax.random.PRNGKey(2))
        assert_trees_are_different(state1, state2)
