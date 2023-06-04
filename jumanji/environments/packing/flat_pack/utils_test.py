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
import pytest

from jumanji.environments.packing.flat_pack.utils import (
    compute_grid_dim,
    get_significant_idxs,
    rotate_block,
)


@pytest.mark.parametrize(
    "num_blocks, expected_grid_dim",
    [
        (1, 3),
        (2, 5),
        (3, 7),
        (4, 9),
        (5, 11),
    ],
)
def test_compute_grid_dim(num_blocks: int, expected_grid_dim: int) -> None:
    """Test that grid dimension is correctly computed given a number of blocks."""
    assert compute_grid_dim(num_blocks) == expected_grid_dim


@pytest.mark.parametrize(
    "grid_dim, expected_idxs",
    [
        (5, jnp.array([2])),
        (7, jnp.array([2, 4])),
        (9, jnp.array([2, 4, 6])),
        (11, jnp.array([2, 4, 6, 8])),
    ],
)
def test_get_significant_idxs(grid_dim: int, expected_idxs: chex.Array) -> None:
    """Test that significant indices are correctly computed given a grid dimension."""
    assert jnp.all(get_significant_idxs(grid_dim) == expected_idxs)


def test_rotate_block(block: chex.Array) -> None:

    # Test with no rotation.
    rotated_block = rotate_block(block, 0)
    assert jnp.array_equal(rotated_block, block)

    # Test 90 degree rotation.
    expected_rotated_block = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    rotated_block = rotate_block(block, 1)
    assert jnp.array_equal(rotated_block, expected_rotated_block)

    # Test 180 degree rotation.
    expected_rotated_block = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    rotated_block = rotate_block(block, 2)
    assert jnp.array_equal(rotated_block, expected_rotated_block)

    # Test 270 degree rotation.
    expected_rotated_block = jnp.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    rotated_block = rotate_block(block, 3)
    assert jnp.array_equal(rotated_block, expected_rotated_block)
