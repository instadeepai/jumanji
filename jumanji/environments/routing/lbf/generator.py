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

from jumanji.environments.routing.lbf.types import Agent, Food, State


# todo: do we need this base class, there is only one viable generator to match lbf?
class Generator(abc.ABC):
    """Base class for generators for the LBF environment."""

    def __init__(
        self,
        grid_size: int,
        num_agents: int,
        num_food: int,
        max_agent_level: int,
    ) -> None:
        """Initialises a LBF generator, used to generate grids for the LBF environment.

        Args:
            grid_size: size of the grid to generate.
            num_agents: number of agents on the grid.
            num_food: number of food items on the grid.
            max_agent_level: maximum level of the agents (inclusive).
            max_food_level: maximum level of the food items (inclusive).
        """
        self._grid_size = grid_size
        self._num_agents = num_agents
        self._num_food = num_food
        self._max_agent_level = max_agent_level

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


class RandomGenerator(Generator):
    """Randomly generates `LBF` grids.

    If too many foods and agents are given for a small grid size, it might not be able to
    place them on the grid. Because of jax this will fail silently and many foods will
    be placed at (0, 0).
    """

    def sample_food(self, key: chex.PRNGKey) -> chex.Array:
        """Samples food positions such that no two foods are adjacent
        and no food is on the edge of the grid.
        """
        flat_size = self.grid_size**2
        pos_keys = jax.random.split(key, self._num_food)

        # cannot place on the edges so mask them out
        mask = jnp.ones(flat_size, dtype=bool)

        top = jnp.arange(self.grid_size)
        bottom = jnp.arange(flat_size - self.grid_size, flat_size)
        left = jnp.arange(0, flat_size, self.grid_size)
        right = jnp.arange(self.grid_size - 1, flat_size, self.grid_size)

        mask = mask.at[top].set(False)
        mask = mask.at[bottom].set(False)
        mask = mask.at[left].set(False)
        mask = mask.at[right].set(False)

        def take_positions(
            mask: chex.Array, key: chex.PRNGKey
        ) -> Tuple[chex.Array, chex.Array]:
            food_pos = jax.random.choice(key=key, a=flat_size, shape=(), p=mask)
            # mask out all adj positions so no foods are placed next to eachother
            adj_positions = jnp.array(
                [
                    food_pos,
                    food_pos + 1,  # right
                    food_pos - 1,  # left
                    food_pos + self.grid_size,  # up
                    food_pos - self.grid_size,  # down
                ]
            )
            return mask.at[adj_positions].set(False), food_pos

        _, food_positions_flat = jax.lax.scan(take_positions, mask, pos_keys)
        food_positions = jnp.divmod(food_positions_flat, self.grid_size)

        return food_positions

    def sample_agents(self, key: chex.PRNGKey, mask: chex.Array) -> chex.Array:
        """Samples agent positions on the grid avoiding the food positions.

        Args:
            key: random key.
            mask: mask of the grid where 1s correspond to empty cells and 0s to food cells.
        """
        positions_flat = jax.random.choice(
            key=key,
            a=self.grid_size**2,
            shape=(self.num_agents,),
            replace=False,  # Player positions cannot overlap
            p=mask,
        )

        return jnp.divmod(positions_flat, self.grid_size)

    def sample_levels(
        self, max_level: int, shape: chex.Shape, key: chex.PRNGKey
    ) -> chex.Array:
        return jax.random.randint(
            key,
            shape=shape,
            minval=1,
            maxval=max_level,
        )

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `LBF` state that contains the grid and the agents' layout.

        Returns:
            A `LBF` state.
        """
        (
            food_pos_key,
            food_level_key,
            agent_pos_key,
            agent_level_key,
            key,
        ) = jax.random.split(key, 5)

        # Generate positions.
        food_positions = self.sample_food(food_pos_key)

        # Place agents on the grid.
        # Mask contains 0's where food is placed, 1's where agents can be placed.
        mask = jnp.ones((self.grid_size, self.grid_size), dtype=bool)
        mask = mask.at[food_positions].set(False)
        mask = mask.ravel()
        agent_positions = self.sample_agents(key=agent_pos_key, mask=mask)

        # Generate levels.
        agent_levels = self.sample_levels(
            self._max_agent_level, (self.num_agents,), agent_level_key
        )
        # In the worst case 3 agents will be needed to eat a food item.
        max_food_level = jnp.sum(agent_levels[:3])
        food_levels = self.sample_levels(
            max_food_level, (self._num_food,), food_level_key
        )

        # Create pytrees for foods and agents.
        agents = jax.vmap(Agent)(
            id=jnp.arange(self.num_agents),
            position=jnp.stack(agent_positions, axis=1),
            level=agent_levels,
        )
        foods = jax.vmap(Food)(
            id=jnp.arange(self._num_food),
            position=jnp.stack(food_positions, axis=1),
            level=food_levels,
        )

        step_count = jnp.array(0, jnp.int32)

        return State(key=key, step_count=step_count, agents=agents, foods=foods)
