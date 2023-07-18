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
from jax import numpy as jnp


class Generator(abc.ABC):
    @property
    @abc.abstractmethod
    def grid_size(self) -> int:
        """Size of the puzzle (n x n grid)."""

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> Tuple[chex.Array, Tuple[int, int]]:
        """Generate a problem instance.

        Args:
            key: jax random key for any stochasticity used in the instance generation process.

        Returns:
            A tuple of a 2D array representing a problem instance and a tuple
            indicating the position of the empty tile.
        """


class RandomGenerator(Generator):
    """A generator for random Sliding Tile Puzzle configurations."""

    def __init__(self, grid_size: int):
        """Initialize the RandomGenerator.

        Args:
            grid_size: The size of the puzzle (n x n grid).
        """
        self._grid_size = grid_size

    @property
    def grid_size(self) -> int:
        return self._grid_size

    def __call__(self, key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
        """Generate a random Sliding Tile Puzzle configuration.

        Args:
            key: PRNGKey used for stochasticity in the generation process.

        Returns:
            A tuple of a 2D array representing a problem instance and a tuple
            indicating the position of the empty tile.
        """
        # Create a list of all tiles
        tiles = jnp.arange(
            self._grid_size * self._grid_size
        )  # create a list of all tiles

        # Shuffle the tiles
        key, subkey = jax.random.split(key)
        shuffled_tiles = jax.random.permutation(subkey, tiles)

        # Reshape the tiles into a 2D array
        puzzle = jnp.reshape(shuffled_tiles, (self._grid_size, self._grid_size))

        # Find the position of the empty tile
        empty_tile_position = jnp.stack(
            jnp.unravel_index(jnp.argmax(puzzle == 0), puzzle.shape)
        )

        return puzzle, empty_tile_position


class SolvableSTPGenerator(Generator):
    """A generator for solvable Sliding Tile Puzzle configurations.

    This generator creates puzzle configurations that are guaranteed to be solvable.
    It starts with a solved configuration
    and makes a series of valid random moves to shuffle the tiles.

    Args:
        grid_size: The size of the puzzle (n x n grid).
        num_shuffle_moves: The number of shuffle moves to perform from the solved state.
    """

    def __init__(self, grid_size: int, num_shuffle_moves: int = 100):
        self._grid_size = grid_size
        self.num_shuffle_moves = num_shuffle_moves

    @property
    def grid_size(self) -> int:
        """Returns the size of the puzzle (n x n grid)."""
        return self._grid_size

    def __call__(self, key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
        """Generate a random Sliding Tile Puzzle configuration.

        Args:
            key: PRNGKey used for stochasticity in the generation process.

        Returns:
            A tuple of a 2D array representing a problem instance and a tuple
            indicating the position of the empty tile.
        """
        # Create a list of all tiles
        tiles = list(range(1, self._grid_size * self._grid_size)) + [
            0
        ]  # The empty tile is represented by 0

        # Shuffle the tiles
        key, subkey = jax.random.split(key)
        shuffled_tiles = jax.random.permutation(subkey, jnp.array(tiles))

        # Reshape the tiles into a 2D array
        puzzle = jnp.reshape(shuffled_tiles, (self._grid_size, self._grid_size))

        # Find the position of the empty tile
        empty_tile_position = jnp.argwhere(puzzle == 0)[0]

        return puzzle, empty_tile_position

    def _random_move(
        self, key: chex.PRNGKey, tiles: chex.Array, empty_tile_pos: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """Performs a valid random move and returns the updated tiles and empty tile position."""
        # Define possible movements
        possible_moves = [
            [-1, 0],  # Up
            [1, 0],  # Down
            [0, -1],  # Left
            [0, 1],  # Right
        ]
        for move in possible_moves:
            new_empty_tile_pos = empty_tile_pos + move
            if (new_empty_tile_pos >= 0).all() and (
                new_empty_tile_pos < self.grid_size
            ).all():
                tiles = self._swap_tiles(tiles, empty_tile_pos, new_empty_tile_pos)
                break
        return tiles, new_empty_tile_pos

    def _swap_tiles(
        self, tiles: chex.Array, pos1: chex.Array, pos2: chex.Array
    ) -> chex.Array:
        """Swaps the tiles at the given positions."""
        flattened_pos1 = pos1[0] * self.grid_size + pos1[1]
        flattened_pos2 = pos2[0] * self.grid_size + pos2[1]

        tiles = tiles.at[flattened_pos1].set(tiles[flattened_pos2])
        tiles = tiles.at[flattened_pos2].set(0)  # Empty tile

        return tiles
