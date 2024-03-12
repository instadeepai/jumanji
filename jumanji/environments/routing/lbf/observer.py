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
from typing import Any, Tuple, Union

import chex
import jax
import jax.numpy as jnp

from jumanji import specs
from jumanji.environments.routing.lbf import utils
from jumanji.environments.routing.lbf.constants import MOVES
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
    Original implementation:
    https://github.com/semitable/lb-foraging/blob/master/lbforaging/foraging/environment.py#L378
    """

    def __init__(
        self, fov: int, grid_size: int, num_agents: int, num_food: int
    ) -> None:
        """
        Initializes the Observer object.

        Args:
            fov (int): The field of view of the agents.
            grid_size (int): The size of the grid.
            num_agents (int): The number of agents in the environment.
            num_food (int): The number of food items in the environment.

        Returns:
            None
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
        self,
        max_agent_level: int,
        max_food_level: int,
        time_limit: int,
    ) -> specs.Spec[Observation]:
        """Returns the observation spec for the environment."""
        pass

    def _action_mask_spec(
        self,
    ) -> specs.BoundedArray:
        """Returns the action mask spec for the environment.

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
    An observer for the LBF environment that provides a vector observation.

    The vector observation is designed based on the structure used in the paper:
    "Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms
    in Cooperative Tasks" - Papoudakis et al.

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

    def transform_positions(
        self, agent: Agent, items: Union[Agent, Food]
    ) -> chex.Array:
        """
        Calculate the positions of items within the agent's field of view.

        Args:
            agent (Agent): The agent whose position is used as the reference point.
            items (Union[Agent, Food]): The items to be transformed.

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
            Tuple[chex.Array, chex.Array, chex.Array]: Arrays positions, and levels.
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
            all_agents (Agent): Containing information about all agents.

        Returns:
            Tuple[chex.Array, chex.Array, chex.Array]: Arrays of position and levels.
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

    def compute_action_mask(self, agent: Agent, state: State) -> chex.Array:
        """
        Calculate the action mask for a given agent based on the current state.

        Args:
            agent (Agent): The agent for which to calculate the action mask.
            state (State): The current state of the environment,
                        containing agent and food information.

        Returns:
            chex.Array: A boolean array representing the action mask for the given agent,
            where `True` indicates a valid action, and `False` indicates an invalid action.
        """

        # Get action mask
        next_positions = agent.position + MOVES

        def check_pos_fn(next_pos: Any, entities: Entity, condition: bool) -> Any:
            return jnp.any(jnp.all(next_pos == entities.position, axis=-1) & condition)

        # Check if any agent is in a next position (condition: The agent doesn't block itself)
        agent_occupied = jax.vmap(check_pos_fn, (0, None, None))(
            next_positions, state.agents, (state.agents.id != agent.id)
        )

        # Check if any food is in a next position (condition: Food must be uneaten)
        food_occupied = jax.vmap(check_pos_fn, (0, None, None))(
            next_positions, state.food_items, ~state.food_items.eaten
        )

        # Check if the next position is out of bounds
        out_of_bounds = jnp.any(
            (next_positions < 0) | (next_positions >= self.grid_size), axis=-1
        )

        action_mask = ~(food_occupied | agent_occupied | out_of_bounds)

        # Check if the agent can load food (only if placed in the neighborhood)
        num_adj_food = (
            jax.vmap(utils.are_entities_adjacent, (0, None))(state.food_items, agent)
            & ~state.food_items.eaten
        )
        is_food_adj = jnp.where(jnp.sum(num_adj_food) > 0, True, False)

        action_mask = jnp.where(is_food_adj, action_mask, action_mask.at[-1].set(False))

        return action_mask

    def make_observation(self, agent: Agent, state: State) -> chex.Array:
        """
        Make an observation for a single agent based on the current state of the environment.

        Args:
            agent (Agent): The agent for which to make the observation and action mask.
        Returns:
            agent_view (chex.Array): The observation for the given agent.
        """

        # Calculate which agents are within the field of view (FOV) of the current agent
        # and are not the current agent itself.
        visible_agents = jnp.all(
            jnp.abs(agent.position - state.agents.position) <= self.fov,
            axis=-1,
        )

        # Calculate which foods are within the FOV of the current agent and are not eaten.
        visible_foods = (
            jnp.all(
                jnp.abs(agent.position - state.food_items.position) <= self.fov,
                axis=-1,
            )
            & ~state.food_items.eaten
        )

        # Placeholder observation for foods and agents
        # this is shown if food or agent is not in view.
        init_vals = jnp.array([-1, -1, 0])
        agent_view = jnp.tile(init_vals, self.num_food + self.num_agents)

        # Extract the positions and levels of visible foods.
        food_xs, food_ys, food_levels = self.extract_foods_info(
            agent, visible_foods, state.food_items
        )

        # Extract the positions and levels of visible agents.
        agent_i_infos, agent_xs, agent_ys, agent_levels = self.extract_agents_info(
            agent, visible_agents, state.agents
        )

        # # Assign the foods and agents infos.
        agent_view = agent_view.at[jnp.arange(0, 3 * self.num_food, 3)].set(food_xs)
        agent_view = agent_view.at[jnp.arange(1, 3 * self.num_food, 3)].set(food_ys)
        agent_view = agent_view.at[jnp.arange(2, 3 * self.num_food, 3)].set(food_levels)

        # # Always place the current agent's info first.
        agent_view = agent_view.at[
            jnp.arange(3 * self.num_food, 3 * self.num_food + 3)
        ].set(agent_i_infos)

        start_idx = 3 * self.num_food + 3
        end_idx = start_idx + 3 * (self.num_agents - 1)
        agent_view = agent_view.at[jnp.arange(start_idx, end_idx, 3)].set(agent_xs)
        agent_view = agent_view.at[jnp.arange(start_idx + 1, end_idx, 3)].set(agent_ys)
        agent_view = agent_view.at[jnp.arange(start_idx + 2, end_idx, 3)].set(
            agent_levels
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
        # Create the observation
        agents_view = jax.vmap(self.make_observation, (0, None))(state.agents, state)

        # Compute the action mask
        action_mask = jax.vmap(self.compute_action_mask, (0, None))(state.agents, state)

        return Observation(
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=state.step_count,
        )

    def observation_spec(
        self,
        max_agent_level: int,
        max_food_level: int,
        time_limit: int,
    ) -> specs.Spec[Observation]:
        """Returns the observation spec for the environment.

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
    An observer for the LBF environment that returns a grid observation.

    This observer provides a grid representation of the environment around each agent.
    The grid is composed of three slices:
    - Agent Slice: Represents the levels of agents.
    - Food Slice: Represents the levels of food items.
    - Access Slice: Indicates empty cells (1) and occupied cells (0).

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

    def state_to_observation(self, state: State) -> Observation:
        """
        Converts a `State` to a grid-based `Observation`.

        Args:
            state (State): The current state of the env, containing agent and food info.

        Returns:
            Observation: An Observation containing agents' views, action masks,
            and step count for all agents.
        """
        # Generate grids with only agents and only foods
        grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
        agent_grids = jax.vmap(utils.place_agent_on_grid, (0, None))(state.agents, grid)
        food_grids = jax.vmap(utils.place_food_on_grid, (0, None))(
            state.food_items, grid
        )

        # Aggregate all agents into one grid and all food into one grid
        agent_grid = jnp.sum(agent_grids, axis=0)
        food_grid = jnp.sum(food_grids, axis=0)

        # Pad the grid to prevent out-of-bounds observation
        agent_grid = jnp.pad(agent_grid, self.fov, constant_values=0)
        food_grid = jnp.pad(food_grid, self.fov, constant_values=0)

        # Get the indexes to slice the grid and obtain the view around the agent
        slice_len = 2 * self.fov + 1, 2 * self.fov + 1
        slice_xs, slice_ys = jax.vmap(utils.slice_around, (0, None))(
            state.agents.position, self.fov
        )

        # Slice agent and food grids to obtain the view around the agent
        agents_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
            agent_grid, (slice_xs, slice_ys), slice_len
        )
        foods_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
            food_grid, (slice_xs, slice_ys), slice_len
        )

        # Compute access mask (action mask in the observation); noop is always available
        access_masks = (agents_view + foods_view) == 0
        access_masks = access_masks.at[:, self.fov, self.fov].set(True)

        # Compute action mask
        # TODO: fix the action mask (load action)
        local_pos = jnp.array([self.fov, self.fov])
        next_local_pos = local_pos + MOVES
        action_mask = access_masks[:, next_local_pos.T[0], next_local_pos.T[1]]

        return Observation(
            agents_view=jnp.stack([agents_view, foods_view, access_masks], axis=1),
            action_mask=action_mask,
            step_count=state.step_count,
        )

    def observation_spec(
        self,
        max_agent_level: int,
        max_food_level: int,
        time_limit: int,
    ) -> specs.Spec[Observation]:
        """Returns the observation spec for the environment.

        Args:
            max_agent_level (int): The maximum level of an agent.
            max_food_level (int): The maximum level of a food.
            time_limit (int): The time limit for the environment.

        Returns:
            specs.Spec[Observation]: The observation spec for the environment.
        """
        max_ob = jnp.max(jnp.array([max_food_level, max_agent_level, self.grid_size]))
        visible_area = 2 * self.fov + 1
        agents_view = specs.BoundedArray(
            shape=(self.num_agents, 3, visible_area, visible_area),
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
