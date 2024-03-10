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
from jumanji.environments.logic.sliding_tile_puzzle.types import State


class Generator(abc.ABC):
    def __init__(self, grid_size: int):
        self._grid_size = grid_size

    @property
    def grid_size(self) -> int:
        """Size of the puzzle (n x n grid)."""
        return self._grid_size

    def make_solved_puzzle(self) -> chex.Array:
        """Creates a solved Sliding Tile Puzzle.

        Returns:
            A solved puzzle.
        """
        return (
            jnp.arange(1, self.grid_size**2 + 1)
            .at[-1]
            .set(EMPTY_TILE)
            .reshape((self.grid_size, self.grid_size))
        )

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> State:
        """Generate a problem instance.

        Args:
            key: jax random key for any stochasticity used in the instance generation process.

        Returns:
            State of the problem instance.
        """


class RandomWalkGenerator(Generator):
    """A Sliding Tile Puzzle generator that samples solvable puzzles using a random walk
    starting from the solved board.

    This generator creates puzzle configurations that are guaranteed to be solvable.
    It starts with a solved configuration and makes a series of valid random moves to shuffle
    the tiles.

    Args:
        grid_size: The size of the puzzle (n x n grid).
        num_random_moves: The number of moves to perform from the solved state.
    """

    def __init__(self, grid_size: int, num_random_moves: int = 100):
        super().__init__(grid_size)
        self.num_random_moves = num_random_moves
        self._solved_puzzle = self.make_solved_puzzle()

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generate a random Sliding Tile Puzzle configuration.

        Args:
            key: PRNGKey used for sampling random actions in the generation process.

        Returns:
            State of the problem instance.
        """
        # Start with the solved puzzle
        puzzle = self._solved_puzzle
        empty_tile_position = jnp.array([self._grid_size - 1, self._grid_size - 1])

        # Perform a number of shuffle moves
        key, moves_key = jax.random.split(key)
        keys = jax.random.split(moves_key, self.num_random_moves)
        (puzzle, empty_tile_position), _ = jax.lax.scan(
            lambda carry, key: (self._make_random_move(key, *carry), None),
            (puzzle, empty_tile_position),
            keys,
        )
        state = State(
            puzzle=puzzle,
            empty_tile_position=empty_tile_position,
            key=key,
            step_count=jnp.zeros((), jnp.int32),
        )
        return state

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

    def _swap_tiles(
        self, puzzle: chex.Array, pos1: chex.Array, pos2: chex.Array
    ) -> chex.Array:
        """Swaps the tiles at the given positions."""
        temp = puzzle[tuple(pos1)]
        puzzle = puzzle.at[tuple(pos1)].set(puzzle[tuple(pos2)])
        puzzle = puzzle.at[tuple(pos2)].set(temp)
        return puzzle
