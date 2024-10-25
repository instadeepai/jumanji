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
import jax.numpy as jnp

from jumanji import specs
from jumanji.environments.routing.lbf import utils
from jumanji.environments.routing.lbf.types import (
    Agent,
    Entity,
    Food,
    Observation,
    State,
)


class LbfObserver(abc.ABC):
    """
    Base class for LBF environment observers.

    The original LBF environment has two different observation types.
    This is a base class to allow for implementing both observation types.
    Original implementation: https://tinyurl.com/make-lbf-obs
    """

    def __init__(self, fov: int, grid_size: int, num_agents: int, num_food: int):
        """
        Initialize the Observer object.

        Args:
            fov (int): The field of view of the agents.
            grid_size (int): The size of the grid.
            num_agents (int): The number of agents in the environment.
            num_food (int): The number of food items in the environment.
        """
        self.fov = fov
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_food = num_food

    @abc.abstractmethod
    def state_to_observation(self, state: State) -> Observation:
        """Converts a `State` to an `Observation`."""
        pass

    @abc.abstractmethod
    def observation_spec(
        self, max_agent_level: int, max_food_level: int, time_limit: int
    ) -> specs.Spec[Observation]:
        """Returns the observation spec for the environment."""
        pass

    def _action_mask_spec(self) -> specs.BoundedArray:
        """
        Returns the action mask spec for the environment.

        The action mask is a boolean array of shape (num_agents, 6). '6' is the number of actions.
        """
        return specs.BoundedArray(
            shape=(self.num_agents, 6),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask",
        )

    def _step_count_spec(self, time_limit: int) -> specs.BoundedArray:
        """Returns the step count spec for the environment."""
        return specs.BoundedArray(
            shape=(),
            dtype=jnp.int32,
            minimum=0,
            maximum=time_limit,
            name="step_count",
        )


class VectorObserver(LbfObserver):
    """
    Provides a vector-based observation of the LBF environment.

     The vector observation is designed based on the structure used in the paper:
     "Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks"
     - Papoudakis et al.

    The observation is a vector of length 3 * (num_food + num_agents) for each agent.
    - The first 3 * num_food elements represent the positions and levels of food items.
    - The next 3 elements indicate the current agent's position and level.
    - The final 3 * (num_agents - 1) elements represent the positions and levels of other agents.

    Foods and agents are represented as (y, x, level). If a food or agent is outside the
    agent's field of view, it is represented as (-1, -1, 0).

    Parameters:
    - fov (int): The field of view of the agents.
    - grid_size (int): The size of the grid.
    - num_agents (int): The number of agents in the environment.
    - num_food (int): The number of food items in the environment.
    """

    def __init__(
        self, fov: int, grid_size: int, num_agents: int, num_food: int
    ) -> None:
        super().__init__(fov, grid_size, num_agents, num_food)

    def transform_positions(self, agent: Agent, items: Entity) -> chex.Array:
        """
        Calculate the positions of items within the agent's field of view.

        Args:
            agent (Agent): The agent whose position is used as the reference point.
            items (Entity): The items to be transformed.

        Returns:
            chex.Array: The transformed positions of the items.
        """
        min_x = jnp.minimum(self.fov, agent.position[0])
        min_y = jnp.minimum(self.fov, agent.position[1])
        return items.position - agent.position + jnp.array([min_x, min_y])

    def extract_foods_info(
        self, agent: Agent, visible_foods: chex.Array, all_foods: Food
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Extract the positions and levels of visible foods.

        Args:
            agent (Agent): The agent observing the foods.
            visible_foods (chex.Array): A boolean array indicating the visibility of foods.
            all_foods (Food): Containing information about all the foods.

        Returns:
            Tuple[chex.Array, chex.Array, chex.Array]: Arrays of positions and levels.
        """
        transformed_positions = self.transform_positions(agent, all_foods)

        food_xs = jnp.where(visible_foods, transformed_positions[:, 0], -1)
        food_ys = jnp.where(visible_foods, transformed_positions[:, 1], -1)
        food_levels = jnp.where(visible_foods, all_foods.level, 0)

        return food_xs, food_ys, food_levels

    def extract_agents_info(
        self, agent: Agent, visible_agents: chex.Array, all_agents: Agent
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Extract the positions and levels of visible agents excluding the current agent.

        Args:
            agent (Agent): The current agent.
            visible_agents (chex.Array): A boolean array indicating the visibility of other agents.
            all_agents (Agent): Containing information on all agents.

        Returns:
            Tuple[chex.Array, chex.Array, chex.Array, chex.Array]: Arrays of positions and levels.
        """
        transformed_positions = self.transform_positions(agent, all_agents)
        agent_xs = jnp.where(visible_agents, transformed_positions[:, 0], -1)
        agent_ys = jnp.where(visible_agents, transformed_positions[:, 1], -1)
        agent_levels = jnp.where(visible_agents, all_agents.level, 0)

        # Remove the current agent's info from all agent's infos.
        agent_i_index = jnp.where(agent.id == all_agents.id, size=1)
        agent_i_infos = jnp.array(
            [
                agent_xs[agent_i_index],
                agent_ys[agent_i_index],
                agent_levels[agent_i_index],
            ]
        ).ravel()

        other_agents_indices = jnp.where(
            agent.id != all_agents.id, size=self.num_agents - 1
        )
        agent_xs = agent_xs[other_agents_indices]
        agent_ys = agent_ys[other_agents_indices]
        agent_levels = agent_levels[other_agents_indices]

        return agent_i_infos, agent_xs, agent_ys, agent_levels

    def make_agents_view(self, agent: Agent, state: State) -> chex.Array:
        """
        Make an observation for a single agent based on the current state of the environment.
        Returns the observation for the given agent."""

        #  Check which agents within in the fov of the current agent.
        visible_agents = jnp.all(
            jnp.abs(agent.position - state.agents.position) <= self.fov, axis=-1
        )

        # Check which food items are visible and are not eaten.
        visible_foods = (
            jnp.all(
                jnp.abs(agent.position - state.food_items.position) <= self.fov,
                axis=-1,
            )
            & ~state.food_items.eaten
        )

        # Placeholder observation.
        init_vals = jnp.array([-1, -1, 0])
        agent_view = jnp.tile(init_vals, self.num_food + self.num_agents)

        food_xs, food_ys, food_levels = self.extract_foods_info(
            agent, visible_foods, state.food_items
        )
        agent_i_infos, agent_xs, agent_ys, agent_levels = self.extract_agents_info(
            agent, visible_agents, state.agents
        )

        # Assign the foods and agents infos.
        agent_view = agent_view.at[jnp.arange(0, 3 * self.num_food, 3)].set(
            food_xs, indices_are_sorted=True, unique_indices=True
        )
        agent_view = agent_view.at[jnp.arange(1, 3 * self.num_food, 3)].set(
            food_ys, indices_are_sorted=True, unique_indices=True
        )
        agent_view = agent_view.at[jnp.arange(2, 3 * self.num_food, 3)].set(
            food_levels, indices_are_sorted=True, unique_indices=True
        )

        # Always place the current agent's info first.
        agent_view = agent_view.at[
            jnp.arange(3 * self.num_food, 3 * self.num_food + 3)
        ].set(agent_i_infos, indices_are_sorted=True, unique_indices=True)

        start_idx = 3 * self.num_food + 3
        end_idx = start_idx + 3 * (self.num_agents - 1)
        agent_view = agent_view.at[jnp.arange(start_idx, end_idx, 3)].set(
            agent_xs, indices_are_sorted=True, unique_indices=True
        )
        agent_view = agent_view.at[jnp.arange(start_idx + 1, end_idx, 3)].set(
            agent_ys, indices_are_sorted=True, unique_indices=True
        )
        agent_view = agent_view.at[jnp.arange(start_idx + 2, end_idx, 3)].set(
            agent_levels, indices_are_sorted=True, unique_indices=True
        )

        return agent_view

    def state_to_observation(self, state: State) -> Observation:
        """
        Convert the current state of the environment into observations for all agents.

        Args:
            state (State): The current state containing agent and food information.

        Returns:
            Observation: An Observation object containing the agents' views, action masks,
                         and step count for all agents.
        """
        # Create the agents' observation.
        agents_view = jax.vmap(self.make_agents_view, (0, None))(state.agents, state)

        # Compute the action mask.
        action_mask = jax.vmap(utils.compute_action_mask, (0, None, None))(
            state.agents, state, self.grid_size
        )

        return Observation(
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=state.step_count,
        )

    def observation_spec(
        self, max_agent_level: int, max_food_level: int, time_limit: int
    ) -> specs.Spec[Observation]:
        """
        Returns the observation spec for the environment.

        Args:
            max_agent_level (int): The maximum level of an agent.
            max_food_level (int): The maximum level of a food.
            time_limit (int): The time limit for the environment.

        Returns:
            specs.Spec[Observation]: The observation spec for the environment.
        """
        max_ob = jnp.max(jnp.array([max_food_level, max_agent_level, self.grid_size]))
        agents_view = specs.BoundedArray(
            shape=(self.num_agents, 3 * (self.num_agents + self.num_food)),
            dtype=jnp.int32,
            name="agents_view",
            minimum=-1,
            maximum=max_ob,
        )

        return specs.Spec(
            Observation,
            "ObservationSpec",
            agents_view=agents_view,
            action_mask=self._action_mask_spec(),
            step_count=self._step_count_spec(time_limit),
        )


class GridObserver(LbfObserver):
    """
    Provides a grid-based observation of the LBF environment.

    Each agent's observation is represented as a three-layer grid:
    - Agent Layer: Indicates the levels of agents within the fov.
    - Food Layer: Shows the levels of visible food items.
    - Accessibility Layer: Marks the accessibility of each cell (1 if empty, 0 if occupied).

    Parameters:
    - fov (int): Field of view view of agents.
    - grid_size (int): Size of the grid.
    - num_agents (int): Total number of agents.
    - num_food (int): Total number of food items.
    """

    def __init__(self, fov: int, grid_size: int, num_agents: int, num_food: int):
        super().__init__(fov, grid_size, num_agents, num_food)

    def make_agents_view(self, state: State) -> chex.Array:
        """Generate grid-based observations for all agents based on the current state."""

        def place_agent_on_grid(agent: Agent, grid: chex.Array) -> chex.Array:
            """Place an agent on the grid."""
            x, y = agent.position
            return grid.at[x + self.fov, y + self.fov].set(agent.level)

        def place_food_on_grid(food: Food, grid: chex.Array) -> chex.Array:
            """Place a food item on the grid."""
            x, y = food.position
            return grid.at[x + self.fov, y + self.fov].set(food.level * ~food.eaten)

        # Initialize grids with extended grid size to prevent out-of-bounds observation
        grid_shape_x_y = self.grid_size + 2 * self.fov
        grid = jnp.zeros((grid_shape_x_y, grid_shape_x_y), dtype=jnp.int32)

        # Place agents and foods on the grid
        agent_grids = jax.vmap(place_agent_on_grid, (0, None))(state.agents, grid)
        agent_grid = jnp.sum(agent_grids, axis=0)
        food_grids = jax.vmap(place_food_on_grid, (0, None))(state.food_items, grid)
        food_grid = jnp.sum(food_grids, axis=0)

        # Create access mask: 1 if cell is accessible else 0.
        access_mask = (agent_grid + food_grid) == 0

        # Account for out-of-bounds by setting the edges to zero
        access_mask = access_mask.at[: self.fov, :].set(0)
        access_mask = access_mask.at[-self.fov :, :].set(0)
        access_mask = access_mask.at[:, : self.fov].set(0)
        access_mask = access_mask.at[:, -self.fov :].set(0)

        # Slice to get local views for each agent
        slice_len = (2 * self.fov + 1, 2 * self.fov + 1)
        agents_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
            agent_grid, state.agents.position, slice_len
        )
        foods_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
            food_grid, state.agents.position, slice_len
        )

        # Slice the access mask similarly
        access_masks = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
            access_mask, state.agents.position, slice_len
        )

        return jnp.stack([agents_view, foods_view, access_masks], axis=1)

    def state_to_observation(self, state: State) -> Observation:
        """
        Converts a `State` to a grid-based `Observation`.
        Returns an `Observation` consisting of grid views, action masks
        and step counts for each agent.
        """

        # Create the agents' observation.
        agents_view = self.make_agents_view(state)

        # Compute the action mask.
        action_mask = jax.vmap(utils.compute_action_mask, (0, None, None))(
            state.agents, state, self.grid_size
        )

        return Observation(
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=state.step_count,
        )

    def observation_spec(
        self, max_agent_level: int, max_food_level: int, time_limit: int
    ) -> specs.Spec[Observation]:
        """
        Returns the observation spec for the environment.

        Args:
            max_agent_level (int): Maximum attainable level for agents.
            max_food_level (int): Maximum quantity of food units an agent can collect.
            time_limit (int): Maximum number of steps per episode.

        Returns:
            specs.Spec[Observation]: The observation spec for the environment.
        """
        max_level = max(max_agent_level, max_food_level, self.grid_size)
        view_dim = 2 * self.fov + 1
        agents_view_spec = specs.BoundedArray(
            shape=(self.num_agents, 3, view_dim, view_dim),
            dtype=jnp.int32,
            minimum=0,
            maximum=max_level,
            name="agents_view",
        )

        return specs.Spec(
            Observation,
            "ObservationSpec",
            agents_view=agents_view_spec,
            action_mask=self._action_mask_spec(),
            step_count=self._step_count_spec(time_limit),
        )
