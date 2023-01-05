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

from jumanji.environments.packing.jobshop.conftest import DummyInstanceGenerator
from jumanji.environments.packing.jobshop.instance_generator import (
    RandomInstanceGenerator,
    ToyInstanceGenerator,
)
from jumanji.environments.packing.jobshop.types import State
from jumanji.testing.pytrees import assert_trees_are_different, assert_trees_are_equal


class TestDummyInstanceGenerator:
    @pytest.fixture
    def dummy_instance_generator(self) -> DummyInstanceGenerator:
        return DummyInstanceGenerator()

    def test_dummy_instance_generator__properties(
        self,
        dummy_instance_generator: DummyInstanceGenerator,
    ) -> None:
        """Validate that the dummy instance generator has the correct properties."""
        assert dummy_instance_generator.num_jobs == 3
        assert dummy_instance_generator.num_machines == 3
        assert dummy_instance_generator.max_num_ops == 3
        assert dummy_instance_generator.max_op_duration == 4

    def test_dummy_instance_generator__call(
        self,
        dummy_instance_generator: ToyInstanceGenerator,
    ) -> None:
        """Validate that the dummy instance generator's call function behaves correctly,
        that it is jittable and compiles only once, and that it returns the same state
        for different keys.
        """
        chex.clear_trace_counter()
        call_fn = jax.jit(
            chex.assert_max_traces(dummy_instance_generator.__call__, n=1)
        )
        state1 = call_fn(jax.random.PRNGKey(1))
        state2 = call_fn(jax.random.PRNGKey(2))
        assert_trees_are_equal(state1, state2)


class TestToyInstanceGenerator:
    @pytest.fixture
    def toy_instance_generator(self) -> ToyInstanceGenerator:
        return ToyInstanceGenerator()

    def test_toy_instance_generator__properties(
        self,
        toy_instance_generator: ToyInstanceGenerator,
    ) -> None:
        """Validate that the toy instance generator has the correct properties."""
        assert toy_instance_generator.num_jobs == 9
        assert toy_instance_generator.num_machines == 5
        assert toy_instance_generator.max_num_ops == 7
        assert toy_instance_generator.max_op_duration == 8

    def test_toy_instance_generator__call(
        self,
        toy_instance_generator: ToyInstanceGenerator,
    ) -> None:
        """Validate that the toy instance generator's call function behaves correctly,
        that it is jittable and compiles only once, and that it returns the same state
        for different keys.
        """
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(toy_instance_generator.__call__, n=1))
        state1 = call_fn(jax.random.PRNGKey(1))
        state2 = call_fn(jax.random.PRNGKey(2))
        assert_trees_are_equal(state1, state2)


class TestRandomInstanceGenerator:
    @pytest.fixture
    def random_instance_generator(self) -> RandomInstanceGenerator:
        return RandomInstanceGenerator(
            num_jobs=20,
            num_machines=10,
            max_num_ops=15,
            max_op_duration=8,
        )

    def test_random_instance_generator__properties(
        self,
        random_instance_generator: RandomInstanceGenerator,
    ) -> None:
        """Validate that the random instance generator has the correct properties."""
        assert random_instance_generator.num_jobs == 20
        assert random_instance_generator.num_machines == 10
        assert random_instance_generator.max_num_ops == 15
        assert random_instance_generator.max_op_duration == 8

    def test_random_instance_generator__call(
        self, random_instance_generator: RandomInstanceGenerator
    ) -> None:
        """Validate that the random instance generator's call function is jittable and compiles
        only once. Also check that giving two different keys results in two different instances.
        """
        chex.clear_trace_counter()
        call_fn = jax.jit(
            chex.assert_max_traces(random_instance_generator.__call__, n=1)
        )
        state1 = call_fn(key=jax.random.PRNGKey(1))
        assert isinstance(state1, State)

        state2 = call_fn(key=jax.random.PRNGKey(2))
        assert_trees_are_different(state1, state2)
