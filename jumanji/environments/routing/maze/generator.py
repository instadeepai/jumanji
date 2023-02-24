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

import chex
import jax.numpy as jnp
from typing_extensions import TypeAlias

from jumanji.environments.commons.maze_utils import maze_generation

Maze: TypeAlias = chex.Array


class Generator(abc.ABC):
    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> Maze:
        """Generate a problem instance.

        Args:
            key: random key.

        Returns:
            A `Maze` representing a problem instance.
        """


class ToyGenerator(Generator):
    """Generate a hardcoded 5x5 toy maze."""

    def __init__(self) -> None:
        self.n_rows = 5
        self.n_cols = 5

    def __call__(self, key: chex.PRNGKey) -> Maze:
        walls = jnp.ones((self.n_rows, self.n_cols), bool)
        walls = walls.at[0, :].set((False, True, False, False, False))
        walls = walls.at[1, :].set((False, True, False, True, True))
        walls = walls.at[2, :].set((False, True, False, False, False))
        walls = walls.at[3, :].set((False, False, False, True, True))
        walls = walls.at[4, :].set((False, False, False, False, False))
        return walls


class RandomGenerator(Generator):
    def __init__(self, rows: int, cols: int) -> None:
        """Random instance generator of the maze environment.

        Args:
            width: the width of the maze to create.
            height: the height of the maze to create.
        """
        self.n_rows = rows
        self.n_cols = cols

    def __call__(self, key: chex.PRNGKey) -> Maze:
        """Generate a random maze.

        This method relies on the `generate_maze` method from the `maze_generation` module to
        generate a maze.

        Args:
            key: the Jax random number generation key.

        Returns:
            maze: A generated maze as an array of booleans.
        """
        return maze_generation.generate_maze(self.n_cols, self.n_rows, key).astype(bool)
