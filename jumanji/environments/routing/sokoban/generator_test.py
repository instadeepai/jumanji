import random

import chex
import jax
import jax.numpy as jnp
import pytest

from jumanji.environments.routing.sokoban.env import Sokoban
from jumanji.environments.routing.sokoban.generator import (
    DeepMindGenerator,
    SimpleSolveGenerator,
    HuggingFaceDeepMindGenerator,
    ToyGenerator,
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

    generator_full=HuggingFaceDeepMindGenerator(
        dataset_name="unfiltered-train",
        proportion_of_files=1,
    )

    assert jnp.array_equal(
        generator_full._fixed_grids.shape,
        (unfiltered_dataset_size, 10, 10),
    )

    generator_half_full=HuggingFaceDeepMindGenerator(
        dataset_name="unfiltered-train",
        proportion_of_files=0.5,
    )

    assert jnp.array_equal(
        generator_half_full._fixed_grids.shape,
        (unfiltered_dataset_size/2, 10, 10),
    )





