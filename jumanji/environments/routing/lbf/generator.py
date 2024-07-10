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

from typing import Tuple

import chex
import jax
from jax import numpy as jnp

from jumanji.environments.routing.lbf.types import Agent, Food, State


class RandomGenerator:
    """Randomly generates `LBF` grids.

    If too many food items and agents are given for a small grid size, it might not be able to
    place them on the grid. Because of jax this will fail silently and many food items will
    be placed at (0, 0).
    """

    def __init__(
        self,
        grid_size: int,
        num_agents: int,
        num_food: int,
        fov: int,
        max_agent_level: int = 2,
        force_coop: bool = False,
    ) -> None:
        """Initialises a LBF generator, used to generate grids for
        the LevelBasedForaging environment.

        Args:
            grid_size: size of the grid to generate.
            fov: field of view of an agent.
            num_agents: number of agents on the grid.
            num_food: number of food items on the grid.
            max_agent_level: maximum level of the agents (inclusive).
            force_coop: Force cooperation between agents.
        """

        if fov is None:
            fov = grid_size
        self._grid_size = grid_size
        self._fov = fov
        self._num_agents = num_agents
        self._num_food = num_food
        self._max_agent_level = max_agent_level
        self._force_coop = force_coop

        # Add assertions to check the validity of the input values.
        assert 5 <= grid_size, "Grid size must be greater than 5."
        assert 2 <= fov <= grid_size, "Field of view must be between 2 and grid_size."
        assert 0 < num_agents, "Number of agents must be positif."
        assert 0 < num_food, "Number of food items must be positif."
        assert (
            max_agent_level >= 2
        ), "Maximum agent level must be equal or greater to 2."

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def fov(self) -> int:
        return self._fov

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def num_food(self) -> int:
        return self._num_food

    @property
    def max_agent_level(self) -> int:
        return self._max_agent_level

    @property
    def force_coop(self) -> int:
        return self._force_coop

    def sample_food(self, key: chex.PRNGKey) -> chex.Array:
        """Randomly samples food positions on the grid, ensuring no two food items are adjacent
        and no food is placed on the edge of the grid.

        Args:
            key (chex.PRNGKey): The random key for reproducible randomness.

        Returns:
            chex.Array: An array containing the flat indices of food on the grid.
                        Each element corresponds to the flattened position of a food item.
        """
        flat_size = self._grid_size**2
        pos_keys = jax.random.split(key, self._num_food)

        # Create a mask to exclude edges
        mask = jnp.ones(flat_size, dtype=bool)
        mask = mask.at[jnp.arange(self._grid_size)].set(False)  # top
        mask = mask.at[jnp.arange(flat_size - self._grid_size, flat_size)].set(
            False
        )  # bottom
        mask = mask.at[jnp.arange(0, flat_size, self._grid_size)].set(False)  # left
        mask = mask.at[jnp.arange(self._grid_size - 1, flat_size, self._grid_size)].set(
            False
        )  # right

        def take_positions(
            mask: chex.Array, key: chex.PRNGKey
        ) -> Tuple[chex.Array, chex.Array]:
            food_flat_pos = jax.random.choice(key=key, a=flat_size, shape=(), p=mask)

            # Mask out adjacent positions to avoid placing food items next to each other
            adj_positions = jnp.array(
                [
                    food_flat_pos,
                    food_flat_pos + 1,  # right
                    food_flat_pos - 1,  # left
                    food_flat_pos + self._grid_size,  # up
                    food_flat_pos - self._grid_size,  # down
                ]
            )

            return mask.at[adj_positions].set(False), food_flat_pos

        _, food_flat_positions = jax.lax.scan(take_positions, mask, pos_keys)

        # Unravel indices to get the 2D coordinates (x, y)
        food_positions_x, food_positions_y = jnp.unravel_index(
            food_flat_positions, (self._grid_size, self._grid_size)
        )
        food_positions = jnp.stack([food_positions_x, food_positions_y], axis=1)

        return food_positions

    def sample_agents(self, key: chex.PRNGKey, mask: chex.Array) -> chex.Array:
        """Randomly samples agent positions on the grid, avoiding positions occupied by food.

        Args:
            key (chex.PRNGKey): The random key.
            mask (chex.Array): The mask of the grid where 1s correspond to empty cells
            and 0s to food cells.

        Returns:
            chex.Array: An array containing the positions of agents on the grid.
                        Each row corresponds to the (x, y) coordinates of an agent.
        """
        agent_flat_positions = jax.random.choice(
            key=key,
            a=self._grid_size**2,
            shape=(self._num_agents,),
            replace=False,  # Avoid agent positions overlaping
            p=mask,
        )
        # Unravel indices to get x and y coordinates
        agent_positions_x, agent_positions_y = jnp.unravel_index(
            agent_flat_positions, (self._grid_size, self._grid_size)
        )

        # Stack x and y coordinates to form a 2D array
        return jnp.stack([agent_positions_x, agent_positions_y], axis=1)

    def sample_levels(
        self, max_level: int, output_shape: chex.Shape, key: chex.PRNGKey
    ) -> chex.Array:
        """Randomly samples levels within the specified shape.

        Args:
            max_level (int): The maximum level (inclusive).
            shape (chex.Shape): The shape of the array to be generated.
            key (chex.PRNGKey): The random key.

        Returns:
            chex.Array: An array containing randomly sampled levels.
        """
        return jax.random.randint(
            key,
            shape=output_shape,
            minval=1,
            maxval=max_level + 1,
        )

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `LBF` state containing the grid and the agents' layout.

        Args:
            key (chex.PRNGKey): The random key for reproducible randomness.

        Returns:
            State: A `LBF` state containing information about the grid, agents, and food items.
        """

        (
            food_pos_key,
            food_level_key,
            agent_pos_key,
            agent_level_key,
            key,
        ) = jax.random.split(key, 5)

        # Generate positions for food items
        food_positions = self.sample_food(food_pos_key)

        # Generate positions for agents. The mask contains 0's where food is placed,
        # 1's where agents can be placed.
        mask = jnp.ones((self._grid_size, self._grid_size), dtype=bool)
        mask = mask.at[food_positions].set(False)
        mask = mask.ravel()
        agent_positions = self.sample_agents(key=agent_pos_key, mask=mask)

        # Generate levels for agents and food items
        agent_levels = self.sample_levels(
            self._max_agent_level, (self._num_agents,), agent_level_key
        )
        max_food_level = jnp.sum(
            jnp.sort(agent_levels)[:3]
        )  # In the worst case, 3 agents are needed to eat a food item

        # Determine food levels based on the maximum level of agents
        food_levels = jnp.where(
            self._force_coop,
            jnp.full(shape=(self._num_food,), fill_value=max_food_level),
            self.sample_levels(max_food_level, (self._num_food,), food_level_key),
        )

        # Create pytrees for agents and food items
        agents = jax.vmap(Agent)(
            id=jnp.arange(self._num_agents),
            position=agent_positions,
            level=agent_levels,
            loading=jnp.zeros((self._num_agents,), dtype=bool),
        )
        food_items = jax.vmap(Food)(
            id=jnp.arange(self._num_food),
            position=food_positions,
            level=food_levels,
            eaten=jnp.zeros((self._num_food,), dtype=bool),
        )
        step_count = jnp.array(0, jnp.int32)

        return State(
            key=key, step_count=step_count, agents=agents, food_items=food_items
        )
