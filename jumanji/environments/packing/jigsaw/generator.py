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

import abc
from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.packing.jigsaw.types import State
from jumanji.environments.packing.jigsaw.utils import (
    compute_grid_dim,
    get_significant_idxs,
    rotate_piece,
)


class InstanceGenerator(abc.ABC):
    """Base class for generators for the jigsaw environment. An `InstanceGenerator` is responsible
    for generating a problem instance when the environment is reset.
    """

    def __init__(
        self,
        num_row_pieces: int,
        num_col_pieces: int,
    ) -> None:
        """Initialises a jigsaw generator, used to generate puzzles for the Jigsaw environment.
        Args:
            num_row_pieces: Number of row pieces in jigsaw puzzle.
            num_col_pieces: Number of column pieces in jigsaw puzzle.
        """

        self.num_row_pieces = num_row_pieces
        self.num_col_pieces = num_col_pieces

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> State:
        """Call method responsible for generating a new state.

        Args:
            key: jax random key in case stochasticity is used in the instance generation process.

        Returns:
            A `JigSaw` environment state.
        """

        raise NotImplementedError


class RandomJigsawGenerator(InstanceGenerator):
    """Random jigsaw generator. This generator will generate a random jigsaw puzzle."""

    def __init__(
        self,
        num_row_pieces: int,
        num_col_pieces: int,
    ):
        """Initialises a random jigsaw generator.
        Args:
            num_row_pieces: Number of row pieces in jigsaw puzzle.
            num_col_pieces: Number of column pieces in jigsaw puzzle.
        """
        super().__init__(num_row_pieces, num_col_pieces)

    def _fill_grid_columns(
        self, carry: Tuple[chex.Array, int], arr_value: int
    ) -> Tuple[Tuple[chex.Array, int], int]:
        """Fills the grid columns with a value.
        This function will fill the grid columns with a value that
        is incremented by 1 each time it is called.
        """
        grid = carry[0]
        grid_x, _ = grid.shape
        fill_value = carry[1]

        fill_value += 1

        edit_grid = jax.lax.dynamic_slice(grid, (0, arr_value), (grid_x, 3))
        edit_grid = jnp.ones_like(edit_grid)
        edit_grid *= fill_value

        grid = jax.lax.dynamic_update_slice(grid, edit_grid, (0, arr_value))

        return (grid, fill_value), arr_value

    def _fill_grid_rows(
        self, carry: Tuple[chex.Array, int, int], arr_value: int
    ) -> Tuple[Tuple[chex.Array, int, int], int]:
        """Fills the grid rows with a value.
        This function will fill the grid rows with a value that
        is incremented by `num_col_pieces` each time it is called.
        """
        grid = carry[0]
        _, grid_y = grid.shape
        sum_value = carry[1]
        num_col_pieces = carry[2]

        edit_grid = jax.lax.dynamic_slice(grid, (arr_value, 0), (3, grid_y))
        edit_grid += sum_value

        sum_value += num_col_pieces

        grid = jax.lax.dynamic_update_slice(grid, edit_grid, (arr_value, 0))

        return (grid, sum_value, num_col_pieces), arr_value

    def _select_sides(self, array: chex.Array, key: chex.PRNGKey) -> chex.Array:
        """Randomly selects a value to replace the center value of an array
        containing three values."""

        selector = jax.random.uniform(key, shape=())

        center_val = jax.lax.cond(
            selector > 0.5,
            lambda: array[0],
            lambda: array[2],
        )

        array = array.at[1].set(center_val)

        return array

    def _select_col_nibs(
        self, carry: Tuple[chex.Array, chex.PRNGKey], col: int
    ) -> Tuple[Tuple[chex.Array, chex.PRNGKey], int]:
        """Creates the nibs for puzzle pieces along columns by randomly selecting
        a value from the left and right side of the column."""

        grid = carry[0]
        key = carry[1]
        rows = grid.shape[0]

        grid_slice = jax.lax.dynamic_slice(grid, (0, col - 1), (rows, 3))
        all_keys = jax.random.split(key, rows + 1)
        key = all_keys[0]
        select_keys = all_keys[1:]
        filled_grid_slice = jax.vmap(self._select_sides)(grid_slice, select_keys)

        grid = jax.lax.dynamic_update_slice(grid, filled_grid_slice, (0, col - 1))

        return (grid, key), col

    def _select_row_nibs(
        self, carry: Tuple[chex.Array, chex.PRNGKey], row: int
    ) -> Tuple[Tuple[chex.Array, chex.PRNGKey], int]:
        """Creates the nibs for puzzle pieces along rows by randomly selecting
        a value from the piece above and below the current piece."""

        grid = carry[0]
        key = carry[1]
        cols = grid.shape[1]

        grid_slice = jax.lax.dynamic_slice(grid, (row - 1, 0), (3, cols))

        grid_slice = grid_slice.T

        all_keys = jax.random.split(key, cols + 1)
        key = all_keys[0]
        select_keys = all_keys[1:]

        filled_grid_slice = jax.vmap(self._select_sides)(grid_slice, select_keys)
        filled_grid_slice = filled_grid_slice.T

        grid = jax.lax.dynamic_update_slice(grid, filled_grid_slice, (row - 1, 0))

        return (grid, key), row

    def _first_nonzero(
        self, arr: chex.Array, axis: int, invalid_val: int = 1000
    ) -> chex.Numeric:
        """Returns the index of the first non-zero value in an array."""
        mask = arr != 0
        return jnp.min(
            jnp.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
        )

    def _crop_nonzero(self, arr_: chex.Array) -> chex.Array:
        """Crops a piece to the size of the piece size."""

        row_roll, col_roll = self._first_nonzero(arr_, axis=0), self._first_nonzero(
            arr_, axis=1
        )

        arr_ = jnp.roll(arr_, -row_roll, axis=0)
        arr_ = jnp.roll(arr_, -col_roll, axis=1)

        cropped_arr = jnp.zeros((3, 3), dtype=jnp.float32)

        cropped_arr = cropped_arr.at[:, :].set(arr_[:3, :3])

        return cropped_arr

    def _extract_piece(
        self, carry: Tuple[chex.Array, chex.PRNGKey], piece_num: int
    ) -> Tuple[Tuple[chex.Array, chex.PRNGKey], chex.Array]:
        """Extracts a puzzle piece from a solved board according to its piece number
        and rotates it by a random amount of degrees."""

        grid, key = carry

        # create a boolean mask for the current piece number
        mask = grid == piece_num
        # use the mask to extract the piece from the grid
        piece = jnp.where(mask, grid, 0)

        # Crop piece
        piece = self._crop_nonzero(piece)

        # Rotate piece by random amount of degrees {0, 90, 180, 270}
        key, rot_key = jax.random.split(key)
        rotation_value = jax.random.randint(key=rot_key, shape=(), minval=0, maxval=4)
        rotated_piece = rotate_piece(piece, rotation_value)

        return (grid, key), rotated_piece

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a random jigsaw puzzle.

        Args:
            key: jax random key in case stochasticity is used in the instance generation process.

        Returns:
            A `JigSaw` environment state.
        """

        num_pieces = self.num_row_pieces * self.num_col_pieces

        # Compute the size of the puzzle board.
        grid_row_dim = compute_grid_dim(self.num_row_pieces)
        grid_col_dim = compute_grid_dim(self.num_col_pieces)

        # Get indices of puzzle where nibs will be placed.
        row_nibs_idxs = get_significant_idxs(grid_row_dim)
        col_nibs_idxs = get_significant_idxs(grid_col_dim)

        # Create an empty puzzle grid.
        grid = jnp.ones((grid_row_dim, grid_col_dim))

        # Fill grid columns with piece numbers
        (grid, _), _ = jax.lax.scan(
            f=self._fill_grid_columns,
            init=(grid, 1),
            xs=col_nibs_idxs,
        )

        # Fill grid rows with piece numbers
        (grid, _, _), _ = jax.lax.scan(
            f=self._fill_grid_rows,
            init=(
                grid,
                self.num_col_pieces,
                self.num_col_pieces,
            ),
            xs=row_nibs_idxs,
        )

        # Create puzzle nibs at relevant rows and columns.
        (grid, key), _ = jax.lax.scan(
            f=self._select_col_nibs, init=(grid, key), xs=col_nibs_idxs
        )

        (solved_board, key), _ = jax.lax.scan(
            f=self._select_row_nibs, init=(grid, key), xs=row_nibs_idxs
        )

        # Extract pieces from the solved board
        _, pieces = jax.lax.scan(
            f=self._extract_piece,
            init=(solved_board, key),
            xs=jnp.arange(1, num_pieces + 1),
        )

        # Finally shuffle the pieces along the leading dimension to
        # untangle a pieces number from its position in the pieces array.
        key, shuffle_pieces_key = jax.random.split(key)
        pieces = jax.random.permutation(
            key=shuffle_pieces_key, x=pieces, axis=0, independent=False
        )

        return State(
            solved_board=solved_board,
            pieces=pieces,
            num_pieces=num_pieces,
            col_nibs_idxs=col_nibs_idxs,
            row_nibs_idxs=row_nibs_idxs,
            action_mask=jnp.ones(
                (num_pieces, 4, grid_row_dim - 2, grid_col_dim - 2), dtype=bool
            ),
            current_board=jnp.zeros_like(solved_board),
            step_count=0,
            key=key,
            placed_pieces=jnp.zeros((num_pieces), dtype=bool),
        )


class ToyJigsawGeneratorWithRotation(InstanceGenerator):
    """Generates a deterministic toy Jigsaw puzzle with 4 pieces. The pieces are
    rotated by a random amount of degrees {0, 90, 180, 270} but not shuffled."""

    def __init__(self) -> None:
        super().__init__(num_row_pieces=2, num_col_pieces=2)

    def __call__(self, key: chex.PRNGKey) -> State:

        del key

        mock_solved_grid = jnp.array(
            [
                [1.0, 1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 2.0, 2.0, 2.0],
                [3.0, 1.0, 4.0, 4.0, 2.0],
                [3.0, 3.0, 4.0, 4.0, 4.0],
                [3.0, 3.0, 3.0, 4.0, 4.0],
            ],
            dtype=jnp.float32,
        )

        mock_pieces = jnp.array(
            [
                [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[2.0, 0.0, 0.0], [2.0, 2.0, 2.0], [2.0, 2.0, 0.0]],
                [[0.0, 0.0, 3.0], [0.0, 3.0, 3.0], [3.0, 3.0, 3.0]],
                [[4.0, 4.0, 0.0], [4.0, 4.0, 4.0], [0.0, 4.0, 4.0]],
            ],
            dtype=jnp.float32,
        )

        return State(
            solved_board=mock_solved_grid,
            pieces=mock_pieces,
            current_board=jnp.zeros_like(mock_solved_grid),
            action_mask=jnp.ones((4, 4, 3, 3), dtype=bool),
            col_nibs_idxs=jnp.array([2], dtype=jnp.int32),
            row_nibs_idxs=jnp.array([2], dtype=jnp.int32),
            num_pieces=jnp.int32(4),
            key=jax.random.PRNGKey(0),
            step_count=0,
            placed_pieces=jnp.zeros(4, dtype=bool),
        )


class ToyJigsawGeneratorNoRotation(InstanceGenerator):
    """Generates a deterministic toy Jigsaw puzzle with 4 pieces. The pieces
    are not rotated and not shuffled."""

    def __init__(self) -> None:
        super().__init__(
            num_row_pieces=2,
            num_col_pieces=2,
        )

    def __call__(self, key: chex.PRNGKey) -> State:

        del key

        mock_solved_grid = jnp.array(
            [
                [1.0, 1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 2.0, 2.0, 2.0],
                [3.0, 1.0, 4.0, 4.0, 2.0],
                [3.0, 3.0, 4.0, 4.0, 4.0],
                [3.0, 3.0, 3.0, 4.0, 4.0],
            ],
            dtype=jnp.float32,
        )

        mock_pieces = jnp.array(
            [
                [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 2.0, 2.0], [2.0, 2.0, 2.0], [0.0, 0.0, 2.0]],
                [[3.0, 0.0, 0.0], [3.0, 3.0, 0.0], [3.0, 3.0, 3.0]],
                [[4.0, 4.0, 0.0], [4.0, 4.0, 4.0], [0.0, 4.0, 4.0]],
            ],
            dtype=jnp.float32,
        )

        return State(
            solved_board=mock_solved_grid,
            pieces=mock_pieces,
            col_nibs_idxs=jnp.array([2], dtype=jnp.int32),
            row_nibs_idxs=jnp.array([2], dtype=jnp.int32),
            num_pieces=jnp.int32(4),
            key=jax.random.PRNGKey(0),
            action_mask=jnp.ones((4, 4, 3, 3), dtype=bool),
            current_board=jnp.zeros_like(mock_solved_grid),
            step_count=0,
            placed_pieces=jnp.zeros(4, dtype=bool),
        )
