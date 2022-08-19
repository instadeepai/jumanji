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
import jax.random
import py
import pytest

from jumanji.binpack.instance_generator import (
    CSVInstanceGenerator,
    RandomInstanceGenerator,
    ToyInstanceGenerator,
    save_instance_to_csv,
)
from jumanji.binpack.types import State
from jumanji.testing.pytrees import assert_trees_are_different, assert_trees_are_equal


def test_save_instance_to_csv(dummy_instance: State, tmpdir: py.path.local) -> None:
    """Validate the dummy instance is correctly saved to a csv file."""
    file_name = "/test.csv"
    save_instance_to_csv(dummy_instance, str(tmpdir.join(file_name)))
    lines = tmpdir.join(file_name).readlines()
    assert lines[0] == "Product_Name,Length,Width,Height,Quantity\n"
    assert lines[1] == "shape_1,1000,700,900,2\n"
    assert lines[2] == "shape_2,500,500,600,1\n"
    assert len(lines) == 3


class TestToyInstanceGenerator:
    @pytest.fixture
    def toy_instance_generator(self) -> ToyInstanceGenerator:
        return ToyInstanceGenerator()

    def test_toy_instance_generator__properties(
        self,
        toy_instance_generator: ToyInstanceGenerator,
    ) -> None:
        """Validate that the toy instance generator has the correct properties."""
        assert toy_instance_generator.max_num_items == 20
        assert toy_instance_generator.max_num_ems > 0

    def test_toy_instance_generator__call(
        self,
        toy_instance_generator: ToyInstanceGenerator,
    ) -> None:
        """Validate that the toy instance generator's call function behaves correctly, that it
        returns the same state for different keys. Also check that it is jittable and compiles only
        once.
        """
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(toy_instance_generator.__call__, n=1))
        state1 = call_fn(jax.random.PRNGKey(1))
        state2 = call_fn(jax.random.PRNGKey(2))
        assert_trees_are_equal(state1, state2)

    def test_toy_instance_generator__generate_solution(
        self,
        toy_instance_generator: ToyInstanceGenerator,
    ) -> None:
        """Validate that the toy instance generator's generate_solution method behaves correctly.
        Also check that it is jittable and compiles only once."""
        state1 = toy_instance_generator(jax.random.PRNGKey(1))

        chex.clear_trace_counter()
        generate_solution = jax.jit(
            chex.assert_max_traces(toy_instance_generator.generate_solution, n=1)
        )

        solution_state1 = generate_solution(jax.random.PRNGKey(1))
        assert isinstance(solution_state1, State)
        assert_trees_are_equal(solution_state1.ems, state1.ems)
        assert_trees_are_different(solution_state1.ems_mask, state1.ems_mask)
        assert_trees_are_equal(solution_state1.items, state1.items)
        assert_trees_are_equal(solution_state1.items_mask, state1.items_mask)
        assert_trees_are_different(solution_state1.items_placed, state1.items_placed)
        assert_trees_are_different(
            solution_state1.items_location, state1.items_location
        )
        assert jnp.all(solution_state1.items_placed)

        solution_state2 = generate_solution(jax.random.PRNGKey(2))
        assert_trees_are_equal(solution_state1, solution_state2)


class TestCSVInstanceGenerator:
    @pytest.fixture
    def csv_instance_generator(
        self, dummy_instance: State, tmpdir: py.path.local, max_num_ems: int = 1
    ) -> CSVInstanceGenerator:
        """Save a dummy instance to a csv file and then use this file to instantiate a
        CSVInstanceGenerator that generates the same dummy instance.
        """
        path = str(tmpdir.join("/for_generator.csv"))
        save_instance_to_csv(dummy_instance, path)
        return CSVInstanceGenerator(path, max_num_ems)

    def test_csv_instance_generator__properties(
        self,
        csv_instance_generator: CSVInstanceGenerator,
    ) -> None:
        """Validate that the csv instance generator has the correct properties."""
        assert csv_instance_generator.max_num_items == 3
        assert csv_instance_generator.max_num_ems == 1

    def test_csv_instance_generator__call(
        self, dummy_instance: State, csv_instance_generator: CSVInstanceGenerator
    ) -> None:
        """Validate that the csv instance generator's call function is jittable and compiles only
        once. Also check that the function is independent of the key.
        """
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(csv_instance_generator.__call__, n=1))
        state1 = call_fn(key=jax.random.PRNGKey(1))
        assert isinstance(state1, State)
        assert_trees_are_equal(state1, dummy_instance)

        state2 = call_fn(key=jax.random.PRNGKey(2))
        assert_trees_are_equal(state1, state2)


class TestRandomInstanceGenerator:
    @pytest.fixture
    def random_instance_generator(
        self, max_num_items: int = 6, max_num_ems: int = 10
    ) -> RandomInstanceGenerator:
        return RandomInstanceGenerator(max_num_items, max_num_ems)

    def test_random_instance_generator__properties(
        self,
        random_instance_generator: RandomInstanceGenerator,
    ) -> None:
        """Validate that the random instance generator has the correct properties."""
        assert random_instance_generator.max_num_items == 6
        assert random_instance_generator.max_num_ems == 10

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

    def test_toy_instance_generator__generate_solution(
        self,
        random_instance_generator: RandomInstanceGenerator,
    ) -> None:
        """Validate that the random instance generator's generate_solution method behaves correctly.
        Also check that it is jittable and compiles only once.
        """
        state1 = random_instance_generator(jax.random.PRNGKey(1))

        chex.clear_trace_counter()
        generate_solution = jax.jit(
            chex.assert_max_traces(random_instance_generator.generate_solution, n=1)
        )

        solution_state1 = generate_solution(jax.random.PRNGKey(1))
        assert isinstance(solution_state1, State)
        assert_trees_are_equal(solution_state1.ems, state1.ems)
        assert_trees_are_different(solution_state1.ems_mask, state1.ems_mask)
        assert_trees_are_equal(solution_state1.items, state1.items)
        assert_trees_are_equal(solution_state1.items_mask, state1.items_mask)
        assert_trees_are_different(solution_state1.items_placed, state1.items_placed)
        assert_trees_are_different(
            solution_state1.items_location, state1.items_location
        )
        assert jnp.all(solution_state1.items_placed | ~solution_state1.items_mask)

        solution_state2 = generate_solution(jax.random.PRNGKey(2))
        assert_trees_are_different(solution_state1, solution_state2)
