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
from jumanji.environments.routing.multi_agent_cleaner.constants import DIRTY, WALL

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


class RandomGenerator(Generator):
    def __init__(self, width: int, height: int) -> None:
        """Random instance generator of the cleaner environment.

        Args:
            width: the width of the maze to create.
            height: the height of the maze to create.
        """
        self.width = width
        self.height = height

    def __call__(self, key: chex.PRNGKey) -> Maze:
        """Generate a random maze.

        This method relies on the `generate_maze` method from the `maze_generation` module to
        generate a maze. This generated maze has its own specific values to represent empty tiles
        and walls. Here, they are replaced respectively with DIRTY and WALL to match the values
        of the cleaner environment.

        Args:
            key: the Jax random number generation key.

        Returns:
            maze: the generated maze.
        """
        maze = maze_generation.generate_maze(self.width, self.height, key)
        return self._adapt_values(maze)

    def _adapt_values(self, maze: Maze) -> Maze:
        """Adapt the values of the maze from maze_generation to agent cleaner."""
        maze = jnp.where(maze == maze_generation.EMPTY, DIRTY, maze)
        # This line currently doesn't do anything, but avoid breaking this function if either of
        # maze_generation.WALL or WALL is changed.
        maze = jnp.where(maze == maze_generation.WALL, WALL, maze)
        return maze
