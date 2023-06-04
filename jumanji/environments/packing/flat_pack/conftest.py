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


@pytest.fixture
def key() -> chex.PRNGKey:
    """A determinstic key."""
    return jax.random.PRNGKey(0)


@pytest.fixture
def block() -> chex.Array:
    return jnp.array(
        [
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )


@pytest.fixture
def solved_grid() -> chex.Array:
    """A mock solved grid for testing."""

    return jnp.array(
        [
            [1.0, 1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0, 2.0],
            [3.0, 1.0, 4.0, 4.0, 2.0],
            [3.0, 3.0, 4.0, 4.0, 4.0],
            [3.0, 3.0, 3.0, 4.0, 4.0],
        ],
    )


@pytest.fixture
def grid_with_block_one_placed() -> chex.Array:
    """A grid with only block one placed."""

    return jnp.array(
        [
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    )


@pytest.fixture()
def block_one_correctly_placed(grid_with_block_one_placed: chex.Array) -> chex.Array:
    """A 2D array of zeros where block one has been placed correctly."""

    return grid_with_block_one_placed


@pytest.fixture()
def block_one_partially_placed(grid_with_block_one_placed: chex.Array) -> chex.Array:
    """A 2D array of zeros where block one has been placed partially correctly.
    That is to say that there is overlap between where the block has been placed and
    where it should be placed to solve the grid."""

    # Shift all elements in the array one down and one to the right
    partially_placed_block = jnp.roll(grid_with_block_one_placed, shift=1, axis=0)
    partially_placed_block = jnp.roll(partially_placed_block, shift=1, axis=1)

    return partially_placed_block
