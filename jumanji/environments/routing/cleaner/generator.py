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
import jax
import jax.numpy as jnp

from jumanji.environments.commons.maze_utils import maze_generation
from jumanji.environments.routing.cleaner.constants import CLEAN, DIRTY, WALL
from jumanji.environments.routing.cleaner.types import State


class Generator(abc.ABC):
    def __init__(self, num_rows: int, num_cols: int, num_agents: int) -> None:
        """Interface for instance generation for the `Cleaner` environment.

        Args:
            num_rows: the width of the grid to create.
            num_cols: the length of the grid to create.
            num_agents: the number of agents.
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_agents = num_agents

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> State:
        """Generate a problem instance for the `Cleaner` environment.

        Args:
            key: random key.

        Returns:
            An initial `State` representing a problem instance.
        """


class RandomGenerator(Generator):
    def __init__(self, num_rows: int, num_cols: int, num_agents: int) -> None:
        super(RandomGenerator, self).__init__(
            num_rows=num_rows, num_cols=num_cols, num_agents=num_agents
        )

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generate a random instance of the cleaner environment.

        This method relies on the `generate_maze` method from the `maze_generation` module to
        generate a maze. This generated maze has its own specific values to represent empty tiles
        and walls. Here, they are replaced respectively with DIRTY and WALL to match the values
        of the cleaner environment.

        Args:
            key: the Jax random number generation key.

        Returns:
            state: the generated state.
        """
        generator_key, state_key = jax.random.split(key)
        maze = maze_generation.generate_maze(
            self.num_cols, self.num_rows, generator_key
        )

        grid = self._adapt_values(maze)

        # Agents start in upper left corner
        agents_locations = jnp.zeros((self.num_agents, 2), jnp.int32)

        # Clean the tile in upper left corner
        grid = grid.at[0, 0].set(CLEAN)

        return State(
            grid=grid,
            agents_locations=agents_locations,
            action_mask=None,
            step_count=jnp.array(0, jnp.int32),
            key=state_key,
        )

    def _adapt_values(self, maze: chex.Array) -> chex.Array:
        """Adapt the values of the maze from maze_generation to agent cleaner."""
        maze = jnp.where(maze == maze_generation.EMPTY, DIRTY, maze)
        # This line currently doesn't do anything, but avoid breaking this function if either of
        # maze_generation.WALL or WALL is changed.
        maze = jnp.where(maze == maze_generation.WALL, WALL, maze)
        return maze
