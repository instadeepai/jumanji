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

from jumanji.environments.packing.flat_pack.generator import RandomFlatPackGenerator


@pytest.fixture
def random_flat_pack_generator() -> RandomFlatPackGenerator:
    """Creates a generator with two row pieces and two column pieces."""
    return RandomFlatPackGenerator(
        num_col_pieces=2,
        num_row_pieces=2,
    )


@pytest.fixture
def grid_only_ones() -> chex.Array:
    """A grid with only ones."""
    return jnp.ones((5, 5))


@pytest.fixture
def grid_columns_partially_filled() -> chex.Array:
    """A grid after one iteration of _fill_grid_columns."""
    # fmt: off
    return jnp.array(
        [
            [1.0, 1.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0, 2.0],
        ]
    )
    # fmt: on


@pytest.fixture
def grid_rows_partially_filled() -> chex.Array:
    """A grid after one iteration of _fill_grid_rows."""
    # fmt: off
    return jnp.array(
        [
            [1.0, 1.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 4.0, 4.0, 4.0],
            [3.0, 3.0, 4.0, 4.0, 4.0],
            [3.0, 3.0, 4.0, 4.0, 4.0],
        ]
    )
    # fmt: on


def test_random_flat_pack_generator__call(
    random_flat_pack_generator: RandomFlatPackGenerator, key: chex.PRNGKey
) -> None:
    """Test that generator generates a valid state."""
    state = random_flat_pack_generator(key)
    assert state.solved_board.shape == (5, 5)
    assert state.num_pieces == 4
    assert state.pieces.shape == (4, 3, 3)
    assert all(state.pieces[i].shape == (3, 3) for i in range(4))
    assert state.col_nibs_idxs == jnp.array([2])
    assert state.row_nibs_idxs == jnp.array([2])
    assert state.action_mask.shape == (4, 4, 3, 3)
    assert state.step_count == 0


def test_random_flat_pack_generator__no_retrace(
    random_flat_pack_generator: RandomFlatPackGenerator, key: chex.PRNGKey
) -> None:
    """Checks that generator call method is only traced once when jitted."""
    keys = jax.random.split(key, 2)
    jitted_generator = jax.jit(
        chex.assert_max_traces((random_flat_pack_generator.__call__), n=1)
    )

    for key in keys:
        jitted_generator(key)


def test_random_flat_pack_generator__fill_grid_columns(
    random_flat_pack_generator: RandomFlatPackGenerator,
    grid_only_ones: chex.Array,
    grid_columns_partially_filled: chex.Array,
) -> None:
    """Checks that _fill_grid_columns method does a single
    step correctly."""

    (grid, fill_value), arr_value = random_flat_pack_generator._fill_grid_columns(
        (grid_only_ones, 1), 2
    )

    assert grid.shape == (5, 5)
    assert jnp.array_equal(grid, grid_columns_partially_filled)
    assert fill_value == 2
    assert arr_value == 2


def test_random_flat_pack_generator__fill_grid_rows(
    random_flat_pack_generator: RandomFlatPackGenerator,
    grid_columns_partially_filled: chex.Array,
    grid_rows_partially_filled: chex.Array,
) -> None:
    """Checks that _fill_grid_columns method does a single
    step correctly."""

    (
        grid,
        sum_value,
        num_col_pieces,
    ), arr_value = random_flat_pack_generator._fill_grid_rows(
        (grid_columns_partially_filled, 2, 2), 2
    )

    assert grid.shape == (5, 5)
    assert jnp.array_equal(grid, grid_rows_partially_filled)
    assert sum_value == 4
    assert num_col_pieces == 2
    assert arr_value == 2


def test_random_flat_pack_generator__select_sides(
    random_flat_pack_generator: RandomFlatPackGenerator, key: chex.PRNGKey
) -> None:
    """Checks that _select_sides method correctly assigns the
    middle value in an array with shape (3,) to either the value
    at index 0 or 2."""

    side_chosen_array = random_flat_pack_generator._select_sides(
        jnp.array([1.0, 2.0, 3.0]), key
    )

    assert side_chosen_array.shape == (3,)
    # check that the output is different from the input
    assert jnp.not_equal(jnp.array([1.0, 2.0, 3.0]), side_chosen_array).any()


def test_random_flat_pack_generator__select_col_nibs(
    random_flat_pack_generator: RandomFlatPackGenerator,
    grid_rows_partially_filled: chex.Array,
    key: chex.PRNGKey,
) -> None:
    """Checks that nibs are created along a given column of the puzzle grid."""

    (
        grid_with_nibs_selected,
        new_key,
    ), column = random_flat_pack_generator._select_col_nibs(
        (grid_rows_partially_filled, key), 2
    )

    assert grid_with_nibs_selected.shape == (5, 5)
    assert jnp.not_equal(key, new_key).all()
    assert column == 2

    selected_col_nibs = grid_with_nibs_selected[:, 2]
    before_selected_nibs_col = grid_rows_partially_filled[:, 2]

    # check that the nibs are different from the column before
    assert jnp.not_equal(selected_col_nibs, before_selected_nibs_col).any()


def test_random_flat_pack_generator__select_row_nibs(
    random_flat_pack_generator: RandomFlatPackGenerator,
    grid_rows_partially_filled: chex.Array,
    key: chex.PRNGKey,
) -> None:
    """Checks that nibs are created along a given row of the puzzle grid."""

    (
        grid_with_nibs_selected,
        new_key,
    ), row = random_flat_pack_generator._select_row_nibs(
        (grid_rows_partially_filled, key), 2
    )

    assert grid_with_nibs_selected.shape == (5, 5)
    assert jnp.not_equal(key, new_key).all()
    assert row == 2

    selected_row_nibs = grid_with_nibs_selected[2, :]
    before_selected_nibs_row = grid_rows_partially_filled[2, :]

    # check that the nibs are different from the row before
    assert jnp.not_equal(selected_row_nibs, before_selected_nibs_row).any()


def test_random_flat_pack_generator__first_nonzero(
    random_flat_pack_generator: RandomFlatPackGenerator,
    piece_one_partially_placed: chex.Array,
) -> None:
    """Checks that the indices of the first non-zero value in a grid is found correctly."""
    first_nonzero_row = random_flat_pack_generator._first_nonzero(
        piece_one_partially_placed, 0
    )
    first_nonzero_col = random_flat_pack_generator._first_nonzero(
        piece_one_partially_placed, 1
    )

    assert first_nonzero_row == 1
    assert first_nonzero_col == 1


def test_random_flat_pack_generator__crop_nonzero(
    random_flat_pack_generator: RandomFlatPackGenerator,
    piece_one_partially_placed: chex.Array,
) -> None:
    """Checks a piece is correctly extracted from a grid of zeros."""
    cropped_piece = random_flat_pack_generator._crop_nonzero(piece_one_partially_placed)

    assert cropped_piece.shape == (3, 3)
    assert jnp.array_equal(
        cropped_piece, jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    )


def test_random_flat_pack_generator__extract_piece(
    random_flat_pack_generator: RandomFlatPackGenerator,
    solved_board: chex.Array,
    key: chex.PRNGKey,
) -> None:
    """Checks that a piece is correctly extracted from a solved puzzle grid."""

    # extract piece number 3
    (_, new_key), piece = random_flat_pack_generator._extract_piece(
        (solved_board, key), 3
    )

    assert piece.shape == (3, 3)
    assert jnp.not_equal(key, new_key).all()
    # check that the piece only contains 3s or 0s
    assert jnp.isin(piece, jnp.array([0.0, 3.0])).all()
