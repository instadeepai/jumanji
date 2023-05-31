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
"""A general utils file for the flat_pack environment."""

import chex
import jax
import jax.numpy as jnp


def compute_grid_dim(num_pieces: int) -> int:
    """Computes the grid dimension given the piece dimension and number of pieces.

    Args:
        num_pieces: The number of puzzle pieces.
    """
    return 3 * num_pieces - (num_pieces - 1)


def get_significant_idxs(grid_dim: int) -> chex.Array:
    """Returns the indices of the grid that are significant. These will be used
    to create puzzle piece nibs.

    Args:
        grid_dim: The dimension of the grid.
    """
    return jnp.arange(grid_dim)[:: 3 - 1][1:-1]


def rotate_piece(piece: chex.Array, rotation_value: int) -> chex.Array:
    """Rotates a piece by {0, 90, 180, 270} degrees.

    Args:
        piece: The piece to rotate.
        rotation: The angle to rotate the piece by.
    """
    rotated_piece = jax.lax.switch(
        index=rotation_value,
        branches=(
            lambda arr: arr,
            lambda arr: jnp.flip(jnp.transpose(arr), axis=1),
            lambda arr: jnp.flip(jnp.flip(arr, axis=0), axis=1),
            lambda arr: jnp.flip(jnp.transpose(arr), axis=0),
        ),
        operand=piece,
    )

    return rotated_piece
