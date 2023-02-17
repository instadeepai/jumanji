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
import jax.numpy as jnp
from typing_extensions import TypeAlias

from jumanji.environments.commons.maze_utils import maze_generation
from jumanji.environments.routing.multi_agent_cleaner.constants import DIRTY, WALL

Maze: TypeAlias = chex.Array


def generate_random_instance(width: int, height: int, key: chex.PRNGKey) -> Maze:
    """Randomly generate an instance of the cleaner environment.

    This method relies on the `generate_maze` method from the `maze_generation` module to generate
    a maze. This generated maze has its own specific values to represent empty tiles and walls.
    Here, they are replaced respectively with DIRTY and WALL to match the values of the cleaner
    environment.

    Args:
        width: the width of the maze to create.
        height: the height of the maze to create.
        key: the Jax random number generation key.

    Returns:
        maze: the generated maze.
    """
    maze = maze_generation.generate_maze(width, height, key)
    maze = maze.at[jnp.where(maze == maze_generation.EMPTY)].set(DIRTY)
    # Adapt the values of walls to this use case.
    maze = maze.at[jnp.where(maze == maze_generation.WALL)].set(WALL)
    return maze
