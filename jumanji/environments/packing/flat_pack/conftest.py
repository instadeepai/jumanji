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
    """A mock block for testing."""

    return jnp.array(
        [
            [0, 1, 1],
            [0, 1, 1],
            [0, 0, 1],
        ]
    )


@pytest.fixture
def solved_grid() -> chex.Array:
    """A mock solved grid for testing."""

    return jnp.array(
        [
            [1, 1, 1, 2, 2],
            [1, 1, 2, 2, 2],
            [3, 1, 4, 4, 2],
            [3, 3, 4, 4, 4],
            [3, 3, 3, 4, 4],
        ],
    )


@pytest.fixture
def grid_with_block_one_placed() -> chex.Array:
    """A grid with only block one placed."""

    return jnp.array(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    )


@pytest.fixture()
def block_one_placed_at_0_0(grid_with_block_one_placed: chex.Array) -> chex.Array:
    """A 2D array of zeros where block one has been placed with it left top-most
    corner at position (0, 0).
    """

    return grid_with_block_one_placed


@pytest.fixture()
def block_one_placed_at_1_1(grid_with_block_one_placed: chex.Array) -> chex.Array:
    """A 2D array of zeros where block one has been placed with it left top-most
    corner at position (1, 1).
    """

    # Shift all elements in the array one down and one to the right
    partially_placed_block = jnp.roll(grid_with_block_one_placed, shift=1, axis=0)
    partially_placed_block = jnp.roll(partially_placed_block, shift=1, axis=1)

    return partially_placed_block


@pytest.fixture()
def action_mask_with_block_1_placed() -> chex.Array:
    """Action mask for a 4 piece grid where only block 1 has been placed with its
    left top-most corner at (1, 1).
    """

    return jnp.array(
        [
            [
                [[False, False, False], [False, False, False], [False, False, False]],
                [[False, False, False], [False, False, False], [False, False, False]],
                [[False, False, False], [False, False, False], [False, False, False]],
                [[False, False, False], [False, False, False], [False, False, False]],
            ],
            [
                [[False, False, True], [False, False, True], [False, True, True]],
                [[False, False, True], [False, True, True], [False, True, True]],
                [[False, False, False], [False, False, True], [True, False, True]],
                [[False, False, False], [False, False, True], [False, False, True]],
            ],
            [
                [[False, False, False], [False, False, True], [True, False, True]],
                [[False, False, False], [False, False, True], [False, False, True]],
                [[False, False, False], [False, False, True], [False, False, True]],
                [[False, False, True], [False, True, True], [True, True, True]],
            ],
            [
                [[False, False, False], [False, False, True], [False, False, True]],
                [[False, False, True], [False, False, True], [False, True, True]],
                [[False, False, False], [False, False, True], [False, False, True]],
                [[False, False, True], [False, False, True], [False, True, True]],
            ],
        ]
    )


@pytest.fixture()
def action_mask_without_only_block_1_placed() -> chex.Array:
    """Action mask for a 4 piece grid where only block 1 can be placed with its
    left top-most corner at (1, 1).
    """

    return jnp.array(
        [
            [
                [[True, False, False], [False, False, False], [False, False, False]],
                [[False, False, False], [False, False, False], [False, False, False]],
                [[False, False, False], [False, False, False], [False, False, False]],
                [[False, False, False], [False, False, False], [False, False, False]],
            ],
            [
                [[False, False, False], [False, False, False], [False, False, False]],
                [[False, False, False], [False, False, False], [False, False, False]],
                [[False, False, False], [False, False, False], [False, False, False]],
                [[False, False, False], [False, False, False], [False, False, False]],
            ],
            [
                [[False, False, False], [False, False, False], [False, False, False]],
                [[False, False, False], [False, False, False], [False, False, False]],
                [[False, False, False], [False, False, False], [False, False, False]],
                [[False, False, False], [False, False, False], [False, False, False]],
            ],
            [
                [[False, False, False], [False, False, False], [False, False, False]],
                [[False, False, False], [False, False, False], [False, False, False]],
                [[False, False, False], [False, False, False], [False, False, False]],
                [[False, False, False], [False, False, False], [False, False, False]],
            ],
        ]
    )
