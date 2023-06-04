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

from jumanji.environments.packing.tetris import utils
from jumanji.environments.packing.tetris.constants import TETROMINOES_LIST
from jumanji.environments.packing.tetris.env import Tetris


@pytest.fixture
def tetris_env() -> Tetris:
    return Tetris(num_rows=6, num_cols=6)


@pytest.fixture
def grid_padded() -> chex.Array:
    """Random grid_padded"""
    grid_padded = jnp.array(
        [
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    return grid_padded


@pytest.fixture
def full_lines() -> chex.Array:
    """Full lines related to the grid_padded"""
    full_lines = jnp.array(
        [False, False, False, False, False, True, False, False, False, False]
    )
    return full_lines


@pytest.fixture
def tetromino() -> chex.Array:
    """Random tetromino"""
    tetromino = jnp.array(
        [
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    return tetromino


def test_clean_lines(grid_padded: chex.Array, full_lines: chex.Array) -> None:
    """Test the jited clean_lines function"""
    new_grid_padded = utils.clean_lines(grid_padded, full_lines)
    assert grid_padded.shape == new_grid_padded.shape
    assert new_grid_padded.sum() + new_grid_padded.shape[1] - 3 == grid_padded.sum()


def test_place_tetromino(grid_padded: chex.Array, tetromino: chex.Array) -> None:
    """Test the jited place_tetromino function"""
    place_tetromino_fn = jax.jit(utils.place_tetromino)
    new_grid_padded, _ = place_tetromino_fn(grid_padded, tetromino, 0)
    cells_count = jnp.clip(new_grid_padded, a_max=1).sum()
    old_cells_count = jnp.clip(grid_padded, a_max=1).sum()
    assert (
        cells_count == old_cells_count + 4
    )  # 4 is the number of filled cells a tetromino
    expected_binary_grid_padded = grid_padded.at[2:6, 0:4].add(tetromino)
    new_grid_padded_binary = jnp.clip(new_grid_padded, a_max=1)
    assert (expected_binary_grid_padded == new_grid_padded_binary).all()


def test_tetromino_action_mask(grid_padded: chex.Array, tetromino: chex.Array) -> None:
    """Test the jited tetromino_action_mask function"""
    action_mask = utils.tetromino_action_mask(grid_padded, tetromino)
    expected_action_mask = jnp.array([True, True, True, False, False, False])
    assert (action_mask == expected_action_mask).all()


def test_sample_tetromino_list() -> None:
    """Test the jited sample_tetromino_list"""
    tetrominoes_list = jnp.array(TETROMINOES_LIST)
    sample_tetromino_list_fn = jax.jit(utils.sample_tetromino_list)
    key = jax.random.PRNGKey(1)
    tetromino, tetromino_index = sample_tetromino_list_fn(key, tetrominoes_list)
    assert tetromino.shape == (4, 4)
    assert tetromino.sum() == 4
    assert tetromino_index in range(len(tetrominoes_list))
