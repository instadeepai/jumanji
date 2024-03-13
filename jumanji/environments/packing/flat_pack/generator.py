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

from jumanji.environments.packing.flat_pack.types import State
from jumanji.environments.packing.flat_pack.utils import (
    compute_grid_dim,
    get_significant_idxs,
    rotate_block,
)


class InstanceGenerator(abc.ABC):
    """Base class for generators for the flat_pack environment. An `InstanceGenerator` is responsible
    for generating a problem instance when the environment is reset.
    """

    def __init__(
        self,
        num_row_blocks: int,
        num_col_blocks: int,
    ) -> None:
        """Initialises a flat_pack generator, used to generate grids for the FlatPack environment.

        Args:
            num_row_blocks: Number of row blocks in flat_pack environment.
            num_col_blocks: Number of column blocks in flat_pack environment.
        """

        self.num_row_blocks = num_row_blocks
        self.num_col_blocks = num_col_blocks

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> State:
        """Call method responsible for generating a new state.

        Args:
            key: jax random key in case stochasticity is used in the instance generation process.

        Returns:
            A `FlatPack` environment state.
        """


class RandomFlatPackGenerator(InstanceGenerator):
    """Random flat_pack generator. This generator will generate a random flat_pack grid."""

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
        is incremented by `num_col_blocks` each time it is called.
        """

        grid = carry[0]
        _, grid_y = grid.shape
        sum_value = carry[1]
        num_col_blocks = carry[2]

        edit_grid = jax.lax.dynamic_slice(grid, (arr_value, 0), (3, grid_y))
        edit_grid += sum_value

        sum_value += num_col_blocks

        grid = jax.lax.dynamic_update_slice(grid, edit_grid, (arr_value, 0))

        return (grid, sum_value, num_col_blocks), arr_value

    def _select_sides(self, array: chex.Array, key: chex.PRNGKey) -> chex.Array:
        """Randomly selects a value to replace the center value of an array
        containing three values.
        """

        selector = jax.random.uniform(key, shape=())

        center_val = jax.lax.cond(
            selector > 0.5,
            lambda: array[0],
            lambda: array[2],
        )

        array = array.at[1].set(center_val)

        return array

    def _select_col_interlocks(
        self, carry: Tuple[chex.Array, chex.PRNGKey], col: int
    ) -> Tuple[Tuple[chex.Array, chex.PRNGKey], int]:
        """Creates interlocks in adjacent blocks along columns by randomly
        selecting a value from the left and right side of the column.
        """

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

    def _select_row_interlocks(
        self, carry: Tuple[chex.Array, chex.PRNGKey], row: int
    ) -> Tuple[Tuple[chex.Array, chex.PRNGKey], int]:
        """Creates interlocks in adjacent blocks along rows by randomly
        selecting a value from the block above and below the current
        block.
        """

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
        """Crops a block to be of shape (3, 3)."""

        row_roll, col_roll = self._first_nonzero(arr_, axis=0), self._first_nonzero(
            arr_, axis=1
        )

        arr_ = jnp.roll(arr_, -row_roll, axis=0)
        arr_ = jnp.roll(arr_, -col_roll, axis=1)

        cropped_arr = jnp.zeros((3, 3), dtype=jnp.int32)

        cropped_arr = cropped_arr.at[:, :].set(arr_[:3, :3])

        return cropped_arr

    def _extract_block(
        self, carry: Tuple[chex.Array, chex.PRNGKey], block_num: int
    ) -> Tuple[Tuple[chex.Array, chex.PRNGKey], chex.Array]:
        """Extracts a block from a solved grid according to its block number
        and rotates it by a random amount of degrees.
        """

        grid, key = carry

        # create a boolean mask for the current block number
        mask = grid == block_num
        # use the mask to extract the block from the grid
        block = jnp.where(mask, grid, 0)

        # Crop block
        block = self._crop_nonzero(block)

        # Rotate block by random amount of degrees {0, 90, 180, 270}
        key, rot_key = jax.random.split(key)
        rotation_value = jax.random.randint(key=rot_key, shape=(), minval=0, maxval=4)
        rotated_block = rotate_block(block, rotation_value)

        return (grid, key), rotated_block

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a random flat_pack grid.

        Args:
            key: jax random key in case stochasticity is used in the instance generation process.

        Returns:
            A `FlatPack` environment state.
        """

        num_blocks = self.num_row_blocks * self.num_col_blocks

        # Compute the size of the grid.
        grid_row_dim = compute_grid_dim(self.num_row_blocks)
        grid_col_dim = compute_grid_dim(self.num_col_blocks)

        # Get indices of grid where interlocks will be.
        row_interlock_idxs = get_significant_idxs(grid_row_dim)
        col_interlock_idxs = get_significant_idxs(grid_col_dim)

        # Create an empty grid.
        grid = jnp.ones((grid_row_dim, grid_col_dim), dtype=jnp.int32)

        # Fill grid columns with block numbers
        (grid, _), _ = jax.lax.scan(
            f=self._fill_grid_columns,
            init=(grid, 1),
            xs=col_interlock_idxs,
        )

        # Fill grid rows with block numbers
        (grid, _, _), _ = jax.lax.scan(
            f=self._fill_grid_rows,
            init=(
                grid,
                self.num_col_blocks,
                self.num_col_blocks,
            ),
            xs=row_interlock_idxs,
        )

        # Create block interlocks at relevant rows and columns.
        (grid, key), _ = jax.lax.scan(
            f=self._select_col_interlocks, init=(grid, key), xs=col_interlock_idxs
        )

        (solved_grid, key), _ = jax.lax.scan(
            f=self._select_row_interlocks, init=(grid, key), xs=row_interlock_idxs
        )

        # Extract blocks from the filled grid
        _, blocks = jax.lax.scan(
            f=self._extract_block,
            init=(solved_grid, key),
            xs=jnp.arange(1, num_blocks + 1),
        )

        # Finally shuffle the blocks along the leading dimension to
        # untangle a block's number from its position in the blocks array.
        key, shuffle_blocks_key = jax.random.split(key)
        blocks = jax.random.permutation(
            key=shuffle_blocks_key, x=blocks, axis=0, independent=False
        )

        return State(
            blocks=blocks,
            num_blocks=num_blocks,
            action_mask=jnp.ones(
                (num_blocks, 4, grid_row_dim - 2, grid_col_dim - 2), dtype=bool
            ),
            grid=jnp.zeros_like(solved_grid),
            step_count=0,
            key=key,
            placed_blocks=jnp.zeros(num_blocks, dtype=bool),
        )


class ToyFlatPackGeneratorWithRotation(InstanceGenerator):
    """Generates a deterministic toy FlatPack environment with 4 blocks. The blocks
    are rotated by a random amount of degrees {0, 90, 180, 270} but not shuffled.
    """

    def __init__(self) -> None:
        super().__init__(num_row_blocks=2, num_col_blocks=2)

    def __call__(self, key: chex.PRNGKey) -> State:

        del key

        solved_grid = jnp.array(
            [
                [1, 1, 1, 2, 2],
                [1, 1, 2, 2, 2],
                [3, 1, 4, 4, 2],
                [3, 3, 4, 4, 4],
                [3, 3, 3, 4, 4],
            ],
            dtype=jnp.int32,
        )

        blocks = jnp.array(
            [
                [[0, 1, 0], [0, 1, 1], [1, 1, 1]],
                [[2, 0, 0], [2, 2, 2], [2, 2, 0]],
                [[0, 0, 3], [0, 3, 3], [3, 3, 3]],
                [[4, 4, 0], [4, 4, 4], [0, 4, 4]],
            ],
            dtype=jnp.int32,
        )

        return State(
            blocks=blocks,
            grid=jnp.zeros_like(solved_grid),
            action_mask=jnp.ones((4, 4, 3, 3), dtype=bool),
            num_blocks=jnp.int32(4),
            key=jax.random.PRNGKey(0),
            step_count=0,
            placed_blocks=jnp.zeros(4, dtype=bool),
        )


class ToyFlatPackGeneratorNoRotation(InstanceGenerator):
    """Generates a deterministic toy FlatPack environment with 4 blocks. The
    blocks are not rotated and not shuffled.
    """

    def __init__(self) -> None:
        super().__init__(num_row_blocks=2, num_col_blocks=2)

    def __call__(self, key: chex.PRNGKey) -> State:

        del key

        solved_grid = jnp.array(
            [
                [1, 1, 1, 2, 2],
                [1, 1, 2, 2, 2],
                [3, 1, 4, 4, 2],
                [3, 3, 4, 4, 4],
                [3, 3, 3, 4, 4],
            ],
            dtype=jnp.int32,
        )

        blocks = jnp.array(
            [
                [[1, 1, 1], [1, 1, 0], [0, 1, 0]],
                [[0, 2, 2], [2, 2, 2], [0, 0, 2]],
                [[3, 0, 0], [3, 3, 0], [3, 3, 3]],
                [[4, 4, 0], [4, 4, 4], [0, 4, 4]],
            ],
            dtype=jnp.int32,
        )

        return State(
            blocks=blocks,
            num_blocks=jnp.int32(4),
            key=jax.random.PRNGKey(0),
            action_mask=jnp.ones((4, 4, 3, 3), dtype=bool),
            grid=jnp.zeros_like(solved_grid),
            step_count=0,
            placed_blocks=jnp.zeros(4, dtype=bool),
        )
