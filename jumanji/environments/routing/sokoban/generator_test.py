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

import random
from typing import List

import chex
import jax
import jax.numpy as jnp
import pytest

from jumanji.environments.routing.sokoban.env import Sokoban
from jumanji.environments.routing.sokoban.generator import (
    DeepMindGenerator,
    HuggingFaceDeepMindGenerator,
)


def test_sokoban__hugging_generator_creation() -> None:
    """checks we can create datasets for all valid boxoban datasets and
    perform a jitted step"""

    datasets = [
        "unfiltered-train",
        "unfiltered-test",
        "unfiltered-valid",
        "medium-train",
        "medium-valid",
        "hard",
    ]

    for dataset in datasets:

        chex.clear_trace_counter()

        env = Sokoban(
            generator=HuggingFaceDeepMindGenerator(
                dataset_name=dataset,
                proportion_of_files=1,
            )
        )

        print(env.generator._fixed_grids.shape)

        step_fn = jax.jit(chex.assert_max_traces(env.step, n=1))

        key = jax.random.PRNGKey(0)
        state, timestep = env.reset(key)

        step_count = 0
        for _ in range(120):
            action = jnp.array(random.randint(0, 4), jnp.int32)
            state, timestep = step_fn(state, action)

            # Check step_count increases after each step
            step_count += 1
            assert state.step_count == step_count
            assert timestep.observation.step_count == step_count


def test_sokoban__hugging_generator_different_problems() -> None:
    """checks that resetting with different keys leads to different problems"""

    chex.clear_trace_counter()

    env = Sokoban(
        generator=HuggingFaceDeepMindGenerator(
            dataset_name="unfiltered-train",
            proportion_of_files=1,
        )
    )

    key1 = jax.random.PRNGKey(0)
    state1, timestep1 = env.reset(key1)

    key2 = jax.random.PRNGKey(1)
    state2, timestep2 = env.reset(key2)

    # Check that different resets lead to different problems
    assert not jnp.array_equal(state2.fixed_grid, state1.fixed_grid)
    assert not jnp.array_equal(state2.variable_grid, state1.variable_grid)


def test_sokoban__hugging_generator_same_problems() -> None:
    """checks that resettting with the same key leads to the same problems"""

    chex.clear_trace_counter()

    env = Sokoban(
        generator=HuggingFaceDeepMindGenerator(
            dataset_name="unfiltered-train",
            proportion_of_files=1,
        )
    )

    key1 = jax.random.PRNGKey(0)
    state1, timestep1 = env.reset(key1)

    key2 = jax.random.PRNGKey(0)
    state2, timestep2 = env.reset(key2)

    assert jnp.array_equal(state2.fixed_grid, state1.fixed_grid)
    assert jnp.array_equal(state2.variable_grid, state1.variable_grid)


def test_sokoban__hugging_generator_proportion_of_problems() -> None:
    """checks that generator initialises correct number of problems"""

    chex.clear_trace_counter()

    unfiltered_dataset_size = 899100

    generator_full = HuggingFaceDeepMindGenerator(
        dataset_name="unfiltered-train",
        proportion_of_files=1,
    )

    assert jnp.array_equal(
        generator_full._fixed_grids.shape,
        (unfiltered_dataset_size, 10, 10),
    )

    generator_half_full = HuggingFaceDeepMindGenerator(
        dataset_name="unfiltered-train",
        proportion_of_files=0.5,
    )

    assert jnp.array_equal(
        generator_half_full._fixed_grids.shape,
        (unfiltered_dataset_size / 2, 10, 10),
    )


def test_sokoban__deepmind_generator_creation() -> None:
    """checks we can create datasets for all valid boxoban datasets and
    perform a jitted step"""

    # Different datasets with varying proportion of files to keep rutime low
    valid_datasets: List[List] = [
        ["unfiltered", "train", 0.01],
        ["unfiltered", "test", 1],
        ["unfiltered", "valid", 0.02],
        ["medium", "train", 0.01],
        ["medium", "valid", 0.02],
        ["hard", None, 1],
    ]

    for dataset in valid_datasets:

        chex.clear_trace_counter()

        env = Sokoban(
            generator=DeepMindGenerator(
                difficulty=dataset[0],
                split=dataset[1],
                proportion_of_files=dataset[2],
            )
        )

        assert env.generator._fixed_grids.shape[0] > 0


def test_sokoban__deepmind_invalid_creation() -> None:
    """checks we can create datasets for all valid boxoban datasets and
    perform a jitted step"""

    # Different datasets with varying proportion of files to keep rutime low
    valid_datasets: List[List] = [
        ["medium", "test", 0.01],
    ]

    for dataset in valid_datasets:

        chex.clear_trace_counter()

        with pytest.raises(Exception):
            _ = Sokoban(
                generator=DeepMindGenerator(
                    difficulty=dataset[0],
                    split=dataset[1],
                    proportion_of_files=dataset[2],
                )
            )
