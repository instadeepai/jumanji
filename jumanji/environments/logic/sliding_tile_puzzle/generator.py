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

from jumanji.environments.logic.sliding_tile_puzzle.constants import EMPTY_TILE, MOVES


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
        tiles = jnp.arange(self._grid_size * self._grid_size)

        # Shuffle the tiles
        key, subkey = jax.random.split(key)
        shuffled_tiles = jax.random.permutation(subkey, tiles)

        # Reshape the tiles into a 2D array
        puzzle = jnp.reshape(shuffled_tiles, (self._grid_size, self._grid_size))

        # Find the position of the empty tile
        empty_tile_position = jnp.stack(
            jnp.unravel_index(jnp.argmax(puzzle == EMPTY_TILE), puzzle.shape)
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
        # Start with a solved puzzle
        puzzle = (
            jnp.arange(1, self._grid_size**2 + 1)
            .at[-1]
            .set(0)
            .reshape((self._grid_size, self._grid_size))
        )

        empty_tile_position = jnp.array([self._grid_size - 1, self._grid_size - 1])

        # Perform a number of shuffle moves
        keys = jax.random.split(key, self.num_shuffle_moves)
        (puzzle, empty_tile_position), _ = jax.lax.scan(
            lambda carry, key: (self._make_random_move(key, *carry), None),
            (puzzle, empty_tile_position),
            keys,
        )

        return puzzle, empty_tile_position

    def _swap_tiles(
        self, puzzle: chex.Array, pos1: chex.Array, pos2: chex.Array
    ) -> chex.Array:
        """Swaps the tiles at the given positions."""
        temp = puzzle[tuple(pos1)]
        puzzle = puzzle.at[tuple(pos1)].set(puzzle[tuple(pos2)])
        puzzle = puzzle.at[tuple(pos2)].set(temp)
        return puzzle

    def _make_random_move(
        self, key: chex.PRNGKey, puzzle: chex.Array, empty_tile_position: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """Makes a random valid move."""
        new_positions = empty_tile_position + MOVES

        # Determine valid moves (known-size boolean array)
        valid_moves_mask = jnp.all(
            (new_positions >= 0) & (new_positions < self._grid_size), axis=-1
        )
        move = jax.random.choice(key, MOVES, shape=(), p=valid_moves_mask)
        new_empty_tile_position = empty_tile_position + move
        # Swap the empty tile with the tile at the new position using _swap_tiles
        updated_puzzle = self._swap_tiles(
            puzzle, empty_tile_position, new_empty_tile_position
        )

        return updated_puzzle, new_empty_tile_position
