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

from jumanji.environments.packing.job_shop.conftest import DummyGenerator
from jumanji.environments.packing.job_shop.generator import (
    MakespanGenerator,
    RandomGenerator,
    ToyGenerator,
)
from jumanji.environments.packing.job_shop.types import State
from jumanji.testing.pytrees import assert_trees_are_different, assert_trees_are_equal


class TestDummyGenerator:
    @pytest.fixture
    def dummy_generator(self) -> DummyGenerator:
        return DummyGenerator()

    def test_dummy_generator__properties(self, dummy_generator: DummyGenerator) -> None:
        """Validate that the dummy instance generator has the correct properties."""
        assert dummy_generator.num_jobs == 3
        assert dummy_generator.num_machines == 3
        assert dummy_generator.max_num_ops == 3
        assert dummy_generator.max_op_duration == 4

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


class TestToyGenerator:
    @pytest.fixture
    def toy_generator(self) -> ToyGenerator:
        return ToyGenerator()

    def test_toy_generator__properties(self, toy_generator: ToyGenerator) -> None:
        """Validate that the toy instance generator has the correct properties."""
        assert toy_generator.num_jobs == 5
        assert toy_generator.num_machines == 4
        assert toy_generator.max_num_ops == 4
        assert toy_generator.max_op_duration == 4

    def test_toy_generator__call(self, toy_generator: ToyGenerator) -> None:
        """Validate that the toy instance generator's call function behaves correctly,
        that it is jit-able and compiles only once, and that it returns the same state
        for different keys.
        """
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(toy_generator.__call__, n=1))
        state1 = call_fn(jax.random.PRNGKey(1))
        state2 = call_fn(jax.random.PRNGKey(2))
        assert_trees_are_equal(state1, state2)


class TestRandomGenerator:
    @pytest.fixture
    def random_generator(self) -> RandomGenerator:
        return RandomGenerator(
            num_jobs=20,
            num_machines=10,
            max_num_ops=15,
            max_op_duration=8,
        )

    def test_random_generator__properties(
        self, random_generator: RandomGenerator
    ) -> None:
        """Validate that the random instance generator has the correct properties."""
        assert random_generator.num_jobs == 20
        assert random_generator.num_machines == 10
        assert random_generator.max_num_ops == 15
        assert random_generator.max_op_duration == 8

    def test_random_generator__call(self, random_generator: RandomGenerator) -> None:
        """Validate that the random instance generator's call function is jit-able and compiles
        only once. Also check that giving two different keys results in two different instances.
        """
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(random_generator.__call__, n=1))
        state1 = call_fn(key=jax.random.PRNGKey(1))
        assert isinstance(state1, State)

        state2 = call_fn(key=jax.random.PRNGKey(2))
        assert_trees_are_different(state1, state2)


class TestMakespanGenerator:
    NUM_JOBS = 30
    NUM_MACHINES = 15
    MAX_NUM_OPS = 10
    MAX_OP_DURATION = 6
    MAKESPAN = 12

    @pytest.fixture
    def makespan_generator(self) -> MakespanGenerator:
        return MakespanGenerator(
            num_jobs=self.NUM_JOBS,
            num_machines=self.NUM_MACHINES,
            max_num_ops=self.MAX_NUM_OPS,
            max_op_duration=self.MAX_OP_DURATION,
            makespan=self.MAKESPAN,
        )

    def test_makespan_generator__attributes(
        self, makespan_generator: MakespanGenerator
    ) -> None:
        """Validate that the random instance generator has the correct properties."""
        assert makespan_generator.num_jobs == self.NUM_JOBS
        assert makespan_generator.num_machines == self.NUM_MACHINES
        assert makespan_generator.max_num_ops == self.MAX_NUM_OPS
        assert makespan_generator.max_op_duration == self.MAX_OP_DURATION

        key = jax.random.PRNGKey(0)
        state = makespan_generator(key)
        assert isinstance(state, State)

    def test_makespan_generator__generate_schedule(
        self, makespan_generator: MakespanGenerator
    ) -> None:
        key = jax.random.PRNGKey(0)
        schedule = makespan_generator._generate_schedule(key)
        assert schedule.shape == (self.NUM_MACHINES, self.MAKESPAN)
