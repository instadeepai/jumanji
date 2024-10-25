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
    """
    Randomly generates Level-Based Foraging (LBF) grids.

    Ensures that no two food items are adjacent and no food is placed on the grid's edge.
    Handles placement of a specified number of agents and food items within a defined grid size.
    """

    def __init__(
        self,
        grid_size: int,
        num_agents: int,
        num_food: int,
        fov: int,
        max_agent_level: int = 2,
        force_coop: bool = False,
    ):
        """
        Initializes the LBF grid generator.

        Args:
            grid_size (int): The size of the grid.
            num_agents (int): The number of agents.
            num_food (int): The number of food items.
            fov (int): Field of view of an agent.
            max_agent_level (int): Maximum level of agents.
            force_coop (bool): Whether to force cooperation among agents.
        """
        assert 5 <= grid_size, "Grid size must be greater than 5."
        assert 1 <= fov <= grid_size, "Field of view must be between 1 and grid_size."
        assert num_agents > 0, "Number of agents must be positive."
        assert num_food > 0, "Number of food items must be positive."
        assert max_agent_level >= 2, "Maximum agent level must be at least 2."

        min_required_cells = num_agents + num_food
        assert (
            grid_size**2
        ) * 0.6 >= min_required_cells, """Ensure at least 40% of the grid cells remain 'unoccupied'
        to facilitate smooth placement and movement of agents and food items."""

        self.grid_size = grid_size
        self.fov = grid_size if fov is None else fov
        self.num_agents = num_agents
        self.num_food = num_food
        self.max_agent_level = max_agent_level
        self.force_coop = force_coop

    def sample_food(self, key: chex.PRNGKey) -> chex.Array:
        """Samples food positions ensuring no 2 are adjacent and none placed on the grid's edge."""

        flat_size = self.grid_size**2
        pos_keys = jax.random.split(key, self.num_food)

        # Create a mask to exclude edges
        mask = jnp.ones(flat_size, dtype=bool)
        mask = mask.at[jnp.arange(self.grid_size)].set(
            False, indices_are_sorted=True, unique_indices=True
        )  # top
        mask = mask.at[jnp.arange(flat_size - self.grid_size, flat_size)].set(
            False, indices_are_sorted=True, unique_indices=True
        )  # bottom
        mask = mask.at[jnp.arange(0, flat_size, self.grid_size)].set(
            False, indices_are_sorted=True, unique_indices=True
        )  # left
        mask = mask.at[jnp.arange(self.grid_size - 1, flat_size, self.grid_size)].set(
            False, indices_are_sorted=True, unique_indices=True
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
                    food_flat_pos + self.grid_size,  # up
                    food_flat_pos - self.grid_size,  # down
                ]
            )

            return mask.at[adj_positions].set(False, unique_indices=True), food_flat_pos

        _, food_flat_positions = jax.lax.scan(take_positions, mask, pos_keys)

        # Unravel indices to get the 2D coordinates (x, y)
        food_positions_x, food_positions_y = jnp.unravel_index(
            food_flat_positions, (self.grid_size, self.grid_size)
        )
        food_positions = jnp.stack([food_positions_x, food_positions_y], axis=1)

        return food_positions

    def sample_agents(self, key: chex.PRNGKey, mask: chex.Array) -> chex.Array:
        """Randomly samples agent positions on the grid, avoiding positions occupied by food.
        Returns an array where each row corresponds to the (x, y) coordinates of an agent.
        """
        agent_flat_positions = jax.random.choice(
            key=key,
            a=self.grid_size**2,
            shape=(self.num_agents,),
            replace=False,  # Avoid agent positions overlaping
            p=mask,
        )
        # Unravel indices to get x and y coordinates
        agent_positions_x, agent_positions_y = jnp.unravel_index(
            agent_flat_positions, (self.grid_size, self.grid_size)
        )

        # Stack x and y coordinates to form a 2D array
        return jnp.stack([agent_positions_x, agent_positions_y], axis=1)

    def sample_levels(
        self, max_level: int, shape: chex.Shape, key: chex.PRNGKey
    ) -> chex.Array:
        """Samples levels within specified bounds."""
        return jax.random.randint(key, shape=shape, minval=1, maxval=max_level + 1)

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a state containing grid, agent, and food item configurations."""
        key_food, key_agents, key_food_level, key_agent_level, key = jax.random.split(
            key, 5
        )

        # Generate positions for food items
        food_positions = self.sample_food(key_food)

        # Generate positions for agents. The mask contains 0's where food is placed,
        # 1's where agents can be placed.
        mask = jnp.ones((self.grid_size, self.grid_size), dtype=bool)
        mask = mask.at[food_positions].set(False)
        mask = mask.ravel()
        agent_positions = self.sample_agents(key=key_agents, mask=mask)

        # Generate levels for agents and food items
        agent_levels = self.sample_levels(
            self.max_agent_level, (self.num_agents,), key_agent_level
        )
        # In the worst case, 3 agents are needed to eat a food item
        max_food_level = jnp.sum(jnp.sort(agent_levels)[:3])

        # Determine food levels based on the maximum level of agents
        food_levels = jnp.where(
            self.force_coop,
            jnp.full(shape=(self.num_food,), fill_value=max_food_level),
            self.sample_levels(max_food_level, (self.num_food,), key_food_level),
        )

        # Create pytrees for agents and food items
        agents = jax.vmap(Agent)(
            id=jnp.arange(self.num_agents),
            position=agent_positions,
            level=agent_levels,
            loading=jnp.zeros((self.num_agents,), dtype=bool),
        )
        food_items = jax.vmap(Food)(
            id=jnp.arange(self.num_food),
            position=food_positions,
            level=food_levels,
            eaten=jnp.zeros((self.num_food,), dtype=bool),
        )
        step_count = jnp.array(0, jnp.int32)

        return State(
            key=key, step_count=step_count, agents=agents, food_items=food_items
        )
