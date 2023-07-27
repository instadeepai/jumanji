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
from typing import Any, Tuple

import chex
import jax
from jax import numpy as jnp

from jumanji.environments.routing.lbf.constants import AGENT, FOOD
from jumanji.environments.routing.lbf.types import Agent, Food, State


class Generator(abc.ABC):
    """Base class for generators for the LBF environment."""

    def __init__(self, grid_size: int, num_agents: int) -> None:
        """Initialises a LBF generator, used to generate grids for the LBF environment.

        Args:
            grid_size: size of the grid to generate.
            num_agents: number of agents on the grid.
        """
        self._grid_size = grid_size
        self._num_agents = num_agents

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `LBF` state that contains the grid and the agents' layout.

        Returns:
            A `LBF` state.
        """


class UniformRandomGenerator(Generator):
    """Randomly generates `LBF` grids that may or may not be solvable. This generator places
    start and target positions uniformly at random on the grid.
    """

    def __init__(
        self,
        grid_size: int,
        num_agents: int,
        num_food: int,
        max_agent_level: int,
        max_food_level: int,
    ) -> None:
        """Instantiates a `UniformRandomGenerator`.

        Args:
            grid_size: size of the square grid to generate.
            num_agents: number of agents/paths on the grid.
        """
        super().__init__(grid_size, num_agents)
        self.num_food = num_food
        self.max_food_level = max_food_level
        self.max_agent_level = max_agent_level

    def sample_food(self, key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
        pos_key, level_key = jax.random.split(key)

        # todo: food positions also cannot be adjacent to eachother
        positions_flat = jax.random.choice(
            key=pos_key,
            a=self.grid_size**2,
            shape=(self.num_food,),
            replace=False,  # Food positions cannot overlap
        )

        levels = jax.random.randint(
            level_key,
            shape=(self.num_food,),
            minval=1,
            maxval=self.max_food_level + 1,
        )

        return jnp.divmod(positions_flat, self.grid_size), levels

    def sample_agents(
        self, key: chex.PRNGKey, mask: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """Samples agent positions on the grid avoiding the food positions.

        Args:
            key: random key.
            mask: mask of the grid where 1s correspond to empty cells and 0s to food cells.
        """
        pos_key, level_key = jax.random.split(key)

        positions_flat = jax.random.choice(
            key=pos_key,
            a=self.grid_size**2,
            shape=(self.num_agents,),
            replace=False,  # Player positions cannot overlap
            p=mask,
        )

        levels = jax.random.randint(
            level_key,
            shape=(self.num_agents,),
            minval=1,
            maxval=self.max_agent_level + 1,
        )

        # sample agent levels
        return jnp.divmod(positions_flat, self.grid_size), levels

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `LBF` state that contains the grid and the agents' layout.

        Returns:
            A `LBF` state.
        """
        # Create empty grid.
        grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)

        food_key, agent_key, key = jax.random.split(key, 3)
        food_positions, food_levels = self.sample_food(key=food_key)

        # Place food on the grid.
        food_ids = jnp.arange(self.num_agents + 1, self.num_agents + self.num_food + 1)
        grid = grid.at[food_positions].set(food_ids)

        # Place agents on the grid.
        # Mask contains 0's where food is placed, 1's where agents can be placed.
        mask = jnp.reshape(grid == 0, -1)
        agent_positions, agent_levels = self.sample_agents(key=agent_key, mask=mask)
        grid = grid.at[agent_positions].set(jnp.arange(1, self.num_agents + 1))

        # Create the agent pytree that corresponds to the grid.
        agents = jax.vmap(Agent)(
            id=jnp.arange(self.num_agents),
            position=jnp.stack(agent_positions, axis=1),
            level=agent_levels,
        )

        # Create the food pytree that corresponds to the grid.
        foods = jax.vmap(Food)(
            id=food_ids,
            position=jnp.stack(food_positions, axis=1),
            level=food_levels,
        )

        step_count = jnp.array(0, jnp.int32)

        return State(
            key=key, grid=grid, step_count=step_count, agents=agents, foods=foods
        )
