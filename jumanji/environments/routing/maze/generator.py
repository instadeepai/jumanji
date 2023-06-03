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
from jumanji.environments.routing.maze.types import Position, State


class Generator(abc.ABC):
    def __init__(self, num_rows: int, num_cols: int):
        """Interface for maze generator.

        Args:
            num_rows: the width of the maze to create.
            num_cols: the length of the maze to create.
        """
        self.num_rows = num_rows
        self.num_cols = num_cols

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> chex.Array:
        """Generate a problem instance.

        Args:
            key: random key.

        Returns:
            state: the generated state.
        """


class ToyGenerator(Generator):
    """Generate a hardcoded 5x5 toy maze."""

    def __init__(self) -> None:
        super(ToyGenerator, self).__init__(num_rows=5, num_cols=5)

    def __call__(self, key: chex.PRNGKey) -> State:
        walls = jnp.ones((self.num_rows, self.num_cols), bool)
        walls = walls.at[0, :].set((False, True, False, False, False))
        walls = walls.at[1, :].set((False, True, False, True, True))
        walls = walls.at[2, :].set((False, True, False, False, False))
        walls = walls.at[3, :].set((False, False, False, True, True))
        walls = walls.at[4, :].set((False, False, False, False, False))

        agent_position = Position(row=0, col=0)
        target_position = Position(row=0, col=4)

        # Build the state.
        return State(
            agent_position=agent_position,
            target_position=target_position,
            walls=walls,
            action_mask=None,
            key=key,
            step_count=jnp.array(0, jnp.int32),
        )


class RandomGenerator(Generator):
    def __init__(self, num_rows: int, num_cols: int) -> None:
        super(RandomGenerator, self).__init__(num_rows=num_rows, num_cols=num_cols)

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generate a random maze.

        This method relies on the `generate_maze` method from the `maze_generation` module to
        generate a maze.

        Args:
            key: the Jax random number generation key.

        Returns:
            state: the generated state.
        """
        key, maze_key, agent_key = jax.random.split(key, 3)

        walls = maze_generation.generate_maze(
            self.num_cols, self.num_rows, maze_key
        ).astype(bool)

        # Randomise agent start and target positions.
        start_and_target_indices = jax.random.choice(
            agent_key,
            jnp.arange(self.num_rows * self.num_cols),
            (2,),
            replace=False,
            p=~walls.flatten(),
        )
        (agent_row, target_row), (agent_col, target_col) = jnp.divmod(
            start_and_target_indices, self.num_cols
        )

        agent_position = Position(row=agent_row, col=agent_col)
        target_position = Position(row=target_row, col=target_col)

        # Build the state.
        return State(
            agent_position=agent_position,
            target_position=target_position,
            walls=walls,
            action_mask=None,
            key=key,
            step_count=jnp.array(0, jnp.int32),
        )
