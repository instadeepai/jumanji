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

from jumanji.environments.logic.minesweeper.constants import UNEXPLORED_ID
from jumanji.environments.logic.minesweeper.types import State
from jumanji.environments.logic.minesweeper.utils import create_flat_mine_locations


class Generator(abc.ABC):
    """Base class for generators for the `Minesweeper` environment."""

    def __init__(self, num_rows: int, num_cols: int, num_mines: int):
        """Initialises a Minesweeper generator for resetting the environment.

        Args:
            num_rows: number of rows, i.e. height of the board.
            num_cols: number of columns, i.e. width of the board.
            num_mines: number of mines to place on the board.
        """
        if num_rows <= 1 or num_cols <= 1:
            raise ValueError(
                f"Should make a board of height and width greater than 1, "
                f"got num_rows={num_rows}, num_cols={num_cols}"
            )
        if num_mines < 0 or num_mines >= num_rows * num_cols:
            raise ValueError(
                f"Number of mines should be constrained between 0 and the size of the board, "
                f"got {num_mines}"
            )
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_mines = num_mines

    @abc.abstractmethod
    def generate_flat_mine_locations(self, key: chex.PRNGKey) -> chex.Array:
        """Generates positions (in flattened coordinates) of the mines in the board."""

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `Minesweeper` state.

        Returns:
            A `Minesweeper` state.
        """
        key, sample_key = jax.random.split(key)
        board = jnp.full(
            shape=(self.num_rows, self.num_cols),
            fill_value=UNEXPLORED_ID,
            dtype=jnp.int32,
        )
        step_count = jnp.array(0, jnp.int32)
        flat_mine_locations = self.generate_flat_mine_locations(key=sample_key)
        state = State(
            board=board,
            step_count=step_count,
            key=key,
            flat_mine_locations=flat_mine_locations,
        )
        return state


class UniformSamplingGenerator(Generator):
    """Generates instances by sampling a given number of mines (without replacement)."""

    def generate_flat_mine_locations(self, key: chex.PRNGKey) -> chex.Array:
        return create_flat_mine_locations(
            key=key,
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            num_mines=self.num_mines,
        )
