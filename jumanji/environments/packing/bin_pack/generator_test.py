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

from jumanji.environments.packing.bin_pack.conftest import DummyGenerator
from jumanji.environments.packing.bin_pack.generator import (
    CSVGenerator,
    ExtendedRandomGenerator,
    ExtendedToyGenerator,
    RandomGenerator,
    ToyGenerator,
    save_instance_to_csv,
)
from jumanji.environments.packing.bin_pack.types import Item, State, item_volume
from jumanji.testing.pytrees import (
    assert_trees_are_close,
    assert_trees_are_different,
    assert_trees_are_equal,
)


def test_save_instance_to_csv(dummy_state: State, tmpdir: py.path.local) -> None:
    """Validate the dummy state is correctly saved to a csv file."""
    file_name = "/test.csv"
    save_instance_to_csv(dummy_state, str(tmpdir.join(file_name)))
    lines = tmpdir.join(file_name).readlines()
    assert lines[0] == "Item_Name,Length,Width,Height,Quantity\n"
    assert lines[1] == "shape_1,1000,700,900,2\n"
    assert lines[2] == "shape_2,500,500,600,1\n"
    assert len(lines) == 3


class TestToyGenerator:
    @pytest.fixture
    def toy_generator(self) -> ToyGenerator:
        return ToyGenerator()

    def test_toy_generator__properties(
        self,
        toy_generator: ToyGenerator,
    ) -> None:
        """Validate that the toy instance generator has the correct properties."""
        assert toy_generator.max_num_items == 20
        assert toy_generator.max_num_ems > 0

    def test_toy_generator__call(
        self,
        toy_generator: ToyGenerator,
    ) -> None:
        """Validate that the toy instance generator's call function behaves correctly, that it
        returns the same state for different keys. Also check that it is jittable and compiles only
        once.
        """
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(toy_generator.__call__, n=1))
        state1 = call_fn(jax.random.PRNGKey(1))
        state2 = call_fn(jax.random.PRNGKey(2))
        assert_trees_are_equal(state1, state2)

    def test_toy_generator__generate_solution(
        self,
        toy_generator: ToyGenerator,
    ) -> None:
        """Validate that the toy instance generator's generate_solution method behaves correctly.
        Also check that it is jittable and compiles only once."""
        state1 = toy_generator(jax.random.PRNGKey(1))

        chex.clear_trace_counter()
        generate_solution = jax.jit(
            chex.assert_max_traces(toy_generator.generate_solution, n=1)
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


class TestCSVGenerator:
    @pytest.fixture
    def csv_generator(
        self,
        dummy_generator: DummyGenerator,
        dummy_state: State,
        tmpdir: py.path.local,
    ) -> CSVGenerator:
        """Save a dummy instance to a csv file and then use this file to instantiate a
        `CSVGenerator` that generates the same dummy instance.
        """
        path = str(tmpdir.join("/for_generator.csv"))
        save_instance_to_csv(dummy_state, path)
        return CSVGenerator(path, dummy_generator.max_num_ems)

    def test_csv_generator__properties(
        self,
        csv_generator: CSVGenerator,
        dummy_generator: DummyGenerator,
    ) -> None:
        """Validate that the csv instance generator has the correct properties."""
        assert csv_generator.max_num_items == dummy_generator.max_num_items
        assert csv_generator.max_num_ems == dummy_generator.max_num_ems

    def test_csv_generator__call(
        self, dummy_state: State, csv_generator: CSVGenerator
    ) -> None:
        """Validate that the csv instance generator's call function is jittable and compiles only
        once. Also check that the function is independent of the key.
        """
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(csv_generator.__call__, n=1))
        state1: State = call_fn(key=jax.random.PRNGKey(1))
        state1.action_mask = jnp.ones(
            (csv_generator.max_num_ems, csv_generator.max_num_items),
            bool,
        )
        assert isinstance(state1, State)
        assert_trees_are_equal(state1, dummy_state)

        state2: State = call_fn(key=jax.random.PRNGKey(2))
        state2.action_mask = jnp.ones(
            (csv_generator.max_num_ems, csv_generator.max_num_items),
            bool,
        )
        assert_trees_are_equal(state1, state2)


class TestRandomGenerator:
    @pytest.fixture
    def random_generator(
        self, max_num_items: int = 6, max_num_ems: int = 10
    ) -> RandomGenerator:
        return RandomGenerator(max_num_items, max_num_ems)

    def test_random_generator__properties(
        self,
        random_generator: RandomGenerator,
    ) -> None:
        """Validate that the random instance generator has the correct properties."""
        assert random_generator.max_num_items == 6
        assert random_generator.max_num_ems == 10

    def test_random_generator__call(self, random_generator: RandomGenerator) -> None:
        """Validate that the random instance generator's call function is jittable and compiles
        only once. Also check that giving two different keys results in two different instances.
        """
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(random_generator.__call__, n=1))
        state1 = call_fn(key=jax.random.PRNGKey(1))
        assert isinstance(state1, State)

        state2 = call_fn(key=jax.random.PRNGKey(2))
        assert_trees_are_different(state1, state2)

    def test_random_generator__generate_solution(
        self,
        random_generator: RandomGenerator,
    ) -> None:
        """Validate that the random instance generator's generate_solution method behaves correctly.
        Also check that it is jittable and compiles only once.
        """
        state1 = random_generator(jax.random.PRNGKey(1))

        chex.clear_trace_counter()
        generate_solution = jax.jit(
            chex.assert_max_traces(random_generator.generate_solution, n=1)
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
        items_volume = (
            item_volume(solution_state1.items) * solution_state1.items_mask
        ).sum()
        assert jnp.isclose(items_volume, solution_state1.container.volume())

        solution_state2 = generate_solution(jax.random.PRNGKey(2))
        assert_trees_are_different(solution_state1, solution_state2)


class TestRandomValueProblemGenerator:
    @pytest.fixture
    def random_generator(
        self, max_num_items: int = 12, max_num_ems: int = 20
    ) -> ExtendedRandomGenerator:
        return ExtendedRandomGenerator(
            max_num_items,
            max_num_ems,
            is_value_based=True,
            is_rotation_allowed=False,
            mean_value=1,
            standard_deviation_value=0.5,
        )

    def test_random_generator__properties(
        self,
        random_generator: ExtendedRandomGenerator,
    ) -> None:
        """Validate that the random instance generator has the correct properties."""
        assert random_generator.max_num_items == 12
        assert random_generator.max_num_ems == 20

    def test_random_generator__call(
        self, random_generator: ExtendedRandomGenerator
    ) -> None:
        """Validate that the random instance generator's call function is jittable and compiles
        only once. Also check that giving two different keys results in two different instances.
        """
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(random_generator.__call__, n=1))
        state1 = call_fn(key=jax.random.PRNGKey(1))
        assert isinstance(state1, State)

        state2 = call_fn(key=jax.random.PRNGKey(2))
        assert_trees_are_different(state1, state2)

    def test_random_generator__generate_solution(
        self,
        random_generator: ExtendedRandomGenerator,
    ) -> None:
        """Validate that the random instance generator's generate_solution method behaves correctly.
        Also check that it is jittable and compiles only once.
        """

        # This will produce a starting state for an environment (no items packed).
        state1 = random_generator(jax.random.PRNGKey(1))

        chex.clear_trace_counter()
        generate_solution = jax.jit(
            chex.assert_max_traces(random_generator.generate_solution, n=1)
        )

        # This will produce a solution to the environment (all possible items packed).
        solution_state1 = generate_solution(jax.random.PRNGKey(1))
        assert isinstance(solution_state1, State)
        assert_trees_are_equal(solution_state1.ems, state1.ems)
        # Should be different because there is only 1 available ems in state1, whereas there should
        # be none available at the end with a solution.
        assert_trees_are_different(solution_state1.ems_mask, state1.ems_mask)
        # The items do not change whether they are packed or not, it is just the items-placed mask
        # that is different.
        assert_trees_are_close(solution_state1.items, state1.items)
        assert_trees_are_equal(solution_state1.items_mask, state1.items_mask)
        # There should be no items placed in state1 whereas half of them will be placed in the
        # perfect solution.
        assert_trees_are_different(solution_state1.items_placed, state1.items_placed)
        # They should be different because state1 has no locations and the solution should have
        # half of them placed.
        assert_trees_are_different(
            solution_state1.items_location, state1.items_location
        )
        # Checks that the perfect solution fills the container exactly
        placed_items_volume = (
            item_volume(solution_state1.items) * solution_state1.items_placed
        ).sum()
        assert jnp.isclose(placed_items_volume, solution_state1.container.volume())

        # Checks that the total volume of all items in an instance is greater than the volume of the
        # container
        items_volume = (
            item_volume(solution_state1.items) * solution_state1.items_mask
        ).sum()
        assert not jnp.isclose(items_volume, solution_state1.container.volume())
        assert items_volume > solution_state1.container.volume()

        # Generates a solution to a new instance (will be new because a different random key).
        solution_state2 = generate_solution(jax.random.PRNGKey(2))
        assert_trees_are_different(solution_state1, solution_state2)


class TestExtendedToyGenerator:
    @pytest.fixture
    def toy_generator(self) -> ExtendedToyGenerator:
        return ExtendedToyGenerator()

    def test_toy_generator__properties(
        self,
        toy_generator: ToyGenerator,
    ) -> None:
        """Validate that the toy instance generator has the correct properties."""
        assert toy_generator.max_num_items == 20
        assert toy_generator.max_num_ems > 0

    def test_toy_generator__call(
        self,
        toy_generator: ToyGenerator,
    ) -> None:
        """Validate that the toy instance generator's call function behaves correctly, that it
        returns the same state for different keys. Also check that it is jittable and compiles only
        once.
        """
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(toy_generator.__call__, n=1))
        state1 = call_fn(jax.random.PRNGKey(1))
        state2 = call_fn(jax.random.PRNGKey(2))
        assert_trees_are_equal(state1, state2)

    def test_toy_generator__generate_solution(
        self,
        toy_generator: ToyGenerator,
    ) -> None:
        """Validate that the toy instance generator's generate_solution method behaves correctly.
        Also check that it is jittable and compiles only once."""
        state1 = toy_generator(jax.random.PRNGKey(1))

        chex.clear_trace_counter()
        generate_solution = jax.jit(
            chex.assert_max_traces(toy_generator.generate_solution, n=1)
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


class TestRotationRandomGenerator:
    @pytest.fixture
    def random_generator(
        self, max_num_items: int = 6, max_num_ems: int = 10
    ) -> RandomGenerator:
        return ExtendedRandomGenerator(
            max_num_items, max_num_ems, is_rotation_allowed=True, is_value_based=False
        )

    def test_random_generator__properties(
        self,
        random_generator: RandomGenerator,
    ) -> None:
        """Validate that the random instance generator has the correct properties."""
        assert random_generator.max_num_items == 6
        assert random_generator.max_num_ems == 10

    def test_random_generator__call(self, random_generator: RandomGenerator) -> None:
        """Validate that the random instance generator's call function is jittable and compiles
        only once. Also check that giving two different keys results in two different instances.
        """
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(random_generator.__call__, n=1))
        state1 = call_fn(key=jax.random.PRNGKey(1))
        assert isinstance(state1, State)

        state2 = call_fn(key=jax.random.PRNGKey(2))
        assert_trees_are_different(state1, state2)

    def test_random_generator__generate_solution(
        self,
        random_generator: RandomGenerator,
    ) -> None:
        """Validate that the random instance generator's generate_solution method behaves correctly.
        Also check that it is jittable and compiles only once.
        """
        state1 = random_generator(jax.random.PRNGKey(1))

        chex.clear_trace_counter()
        generate_solution = jax.jit(
            chex.assert_max_traces(random_generator.generate_solution, n=1)
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
        assert jnp.all(solution_state1.items_placed[0] | ~solution_state1.items_mask)
        non_rotated_items = Item(
            solution_state1.items.x_len[0],
            solution_state1.items.y_len[0],
            solution_state1.items.z_len[0],
        )
        items_volume = (
            item_volume(non_rotated_items) * solution_state1.items_mask[0]
        ).sum()
        assert jnp.isclose(items_volume, solution_state1.container.volume())

        solution_state2 = generate_solution(jax.random.PRNGKey(2))
        assert_trees_are_different(solution_state1, solution_state2)


class TestExtendedRandomGenerator:
    @pytest.fixture
    def random_generator(
        self, max_num_items: int = 12, max_num_ems: int = 20
    ) -> RandomGenerator:
        return ExtendedRandomGenerator(
            max_num_items,
            max_num_ems,
            is_rotation_allowed=True,
            is_value_based=True,
            mean_value=1,
            standard_deviation_value=0.5,
        )

    def test_random_generator__properties(
        self,
        random_generator: RandomGenerator,
    ) -> None:
        """Validate that the random instance generator has the correct properties."""
        assert random_generator.max_num_items == 12
        assert random_generator.max_num_ems == 20

    def test_random_generator__call(self, random_generator: RandomGenerator) -> None:
        """Validate that the random instance generator's call function is jittable and compiles
        only once. Also check that giving two different keys results in two different instances.
        """
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(random_generator.__call__, n=1))
        state1 = call_fn(key=jax.random.PRNGKey(1))
        assert isinstance(state1, State)

        state2 = call_fn(key=jax.random.PRNGKey(2))
        assert_trees_are_different(state1, state2)

    def test_random_generator__generate_solution(
        self,
        random_generator: RandomGenerator,
    ) -> None:
        """Validate that the random instance generator's generate_solution method behaves correctly.
        Also check that it is jittable and compiles only once.
        """
        state1 = random_generator(jax.random.PRNGKey(1))

        chex.clear_trace_counter()
        generate_solution = jax.jit(
            chex.assert_max_traces(random_generator.generate_solution, n=1)
        )

        solution_state1 = generate_solution(jax.random.PRNGKey(1))
        assert isinstance(solution_state1, State)
        assert_trees_are_equal(solution_state1.ems, state1.ems)
        assert_trees_are_different(solution_state1.ems_mask, state1.ems_mask)
        assert_trees_are_close(solution_state1.items, state1.items)
        assert_trees_are_equal(solution_state1.items_mask, state1.items_mask)
        assert_trees_are_different(solution_state1.items_placed, state1.items_placed)
        assert_trees_are_different(
            solution_state1.items_location, state1.items_location
        )

        placed_items_volume = (
            item_volume(solution_state1.items) * solution_state1.items_placed
        ).sum()
        assert jnp.isclose(placed_items_volume, solution_state1.container.volume())

        # Checks that the total volume of all items in an instance is greater than the volume of the
        # container
        items_volume = (
            item_volume(solution_state1.items) * solution_state1.items_mask
        ).sum()
        assert not jnp.isclose(items_volume, solution_state1.container.volume())
        assert items_volume > solution_state1.container.volume()

        # Generates a solution to a new instance (will be new because a different random key).
        solution_state2 = generate_solution(jax.random.PRNGKey(2))
        assert_trees_are_different(solution_state1, solution_state2)
