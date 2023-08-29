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

from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp

import jumanji.environments.routing.lbf.utils as utils
from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.lbf.constants import LOAD, MOVES
from jumanji.environments.routing.lbf.generator import UniformRandomGenerator
from jumanji.environments.routing.lbf.types import Agent, Food, Observation, State
from jumanji.types import TimeStep, restart, termination, transition


class LevelBasedForaging(Environment[State]):
    def __init__(
        self,
        generator: Optional[UniformRandomGenerator] = None,
        fov: int = 10,
        time_limit: int = 50,
    ) -> None:
        super().__init__()

        self._generator = generator or UniformRandomGenerator(
            grid_size=10, num_agents=3, num_food=3, max_agent_level=2, max_food_level=6
        )
        self._fov = fov
        self._time_limit = time_limit
        self.time_limit = time_limit
        self.num_agents = self._generator.num_agents
        self.num_obs_features = self.num_agents * 3 + self._generator.num_food * 3

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        state = self._generator(key)
        observation = self._state_to_obs(state)

        return state, restart(observation, shape=self._generator.num_agents)

    def step(self, state: State, actions: chex.Array) -> Tuple[State, TimeStep]:
        # move agents, fix collisions that may happen and set loading status
        moved_agents = jax.vmap(utils.move, (0, 0, None, None))(
            state.agents,
            actions,
            state.foods,
            self._generator.grid_size,
        )
        # check that no two agent share the same position after moving
        moved_agents = utils.fix_collisions(moved_agents, state.agents)

        # set agent's loading status
        moved_agents = jax.vmap(
            lambda agent, action: agent.replace(loading=action == LOAD)
        )(moved_agents, actions)

        # eat food
        foods, eaten, adj_loading_level = jax.vmap(utils.eat, (None, 0))(
            moved_agents, state.foods
        )

        reward = self.get_reward(foods, adj_loading_level, eaten)

        state = State(
            agents=moved_agents,
            foods=foods,
            step_count=state.step_count + 1,
            key=state.key,
        )

        observation = self._state_to_obs(state)
        # First condition is truncation, second is termination.
        # Jumanji doesn't support truncation yet...
        done = (state.step_count >= self._time_limit) | jnp.all(state.foods.eaten)
        timestep = jax.lax.cond(
            done,
            lambda: termination(reward, observation, shape=self._generator.num_agents),
            lambda: transition(reward, observation, shape=self._generator.num_agents),
        )

        return state, timestep

    # Do we make this a param of the env even though there's probably only a single type of reward?
    def get_reward(
        self, foods: Food, adj_agent_levels: chex.Array, eaten: chex.Array
    ) -> chex.Array:
        """Returns a reward for all agents given all foods."""
        # Get reward per food for all food (by vmapping over foods).
        # Then sum that reward on agent dim to get reward per agent.
        return jnp.sum(
            jax.vmap(self._reward_per_food, in_axes=(0, 0, 0, None))(
                foods, adj_agent_levels, eaten, jnp.sum(foods.level)
            ),
            axis=(0),
        )

    def _reward_per_food(
        self,
        food: Food,
        adj_agent_levels: chex.Array,
        eaten: chex.Array,
        total_food_level: chex.Array,
    ) -> chex.Array:
        """Returns the reward for all agents given a single food.

        Args:
            agents: all the agents in the environment.
            food: a food that may or may not have been eaten.
            adj_agent_levels: the level of the agent adjacent to the food,
             this is 0 if the agent is not adjacent.
        """
        total_adj_level = jnp.sum(adj_agent_levels)
        # zero out all agents if food was not eaten
        adj_levels_if_eaten = adj_agent_levels * eaten

        # todo: think this can be done through normal broadcasting
        def _reward(adj_level_if_eaten: chex.Numeric, food: Food) -> chex.Array:
            """Returns the reward for a single agent given it's level if it was adjacent."""
            reward = adj_level_if_eaten * food.level
            normalizer = total_adj_level * total_food_level
            # It's often the case that no agents are adjacent to the food
            # so we need to avoid dividing by 0 -> nan_to_num
            return jnp.nan_to_num(reward / normalizer)

        return jax.vmap(_reward, (0, None))(adj_levels_if_eaten, food)

    def _state_to_obs(self, state: State) -> Observation:
        # get grids with only agents and grid with only foods
        # obs_size = 3 * self._generator.num_agents + 3 * self._generator.num_food
        num_agents = self._generator.num_agents

        def make_obs(agent: Agent):
            dist = jnp.array([self._fov, self._fov])

            neighbour_agents = jnp.all(
                jnp.abs(agent.position - state.agents.position) <= dist
            ) & ~(agent.id == state.agents.id)
            # neighbour_agents = (
            #     jnp.linalg.norm(agent.position - state.agents.position, axis=-1)
            #     <= jnp.sqrt(2)
            # ) & ~(agent.id == state.agents.id)

            # neighbour_foods = (
            #     jnp.linalg.norm(agent.position - state.foods.position, axis=-1)
            #     <= jnp.sqrt(2)
            # ) & ~state.foods.eaten
            neighbour_foods = (
                jnp.all(jnp.abs(agent.position - state.foods.position) <= dist)
                & ~state.foods.eaten
            )

            num_food = self._generator.num_food
            init_vals = jnp.array([-1, -1, 0])
            obs = jnp.tile(init_vals, num_food + num_agents)
            food_ys = jnp.where(neighbour_foods, state.foods.position[:, 0], -1)
            food_xs = jnp.where(neighbour_foods, state.foods.position[:, 1], -1)
            food_levels = jnp.where(neighbour_foods, state.foods.level, 0)

            agent_ys = jnp.where(neighbour_agents, state.agents.position[:, 0], -1)
            agent_xs = jnp.where(neighbour_agents, state.agents.position[:, 1], -1)
            agent_levels = jnp.where(neighbour_agents, state.agents.level, 0)

            # filter
            agent_ys_i = jnp.where(agent.id != state.agents.id, size=num_agents - 1)
            agent_xs_i = jnp.where(agent.id != state.agents.id, size=num_agents - 1)
            agent_levels_i = jnp.where(agent.id != state.agents.id, size=num_agents - 1)
            agent_ys = agent_ys[agent_ys_i]
            agent_xs = agent_xs[agent_xs_i]
            agent_levels = agent_levels[agent_levels_i]

            obs = obs.at[jnp.arange(0, 3 * num_food, 3)].set(food_ys)
            obs = obs.at[jnp.arange(1, 3 * num_food, 3)].set(food_xs)
            obs = obs.at[jnp.arange(2, 3 * num_food, 3)].set(food_levels)

            # current agent always first
            obs = obs.at[3 * num_food].set(agent.position[0])
            obs = obs.at[3 * num_food + 1].set(agent.position[1])
            obs = obs.at[3 * num_food + 2].set(agent.level)

            agent_start_idx = 3 * num_food + 3
            obs = obs.at[
                jnp.arange(agent_start_idx, agent_start_idx + 3 * (num_agents - 1), 3)
            ].set(agent_ys)
            obs = obs.at[
                jnp.arange(
                    agent_start_idx + 1, agent_start_idx + 3 * (num_agents - 1), 3
                )
            ].set(agent_xs)
            obs = obs.at[
                jnp.arange(
                    agent_start_idx + 2, agent_start_idx + 3 * (num_agents - 1), 3
                )
            ].set(agent_levels)

            return obs

        # other method - gets the action mask
        grid_size = self._generator.grid_size
        grid = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
        agent_grids = jax.vmap(utils.place_agent_on_grid, (0, None))(state.agents, grid)
        food_grids = jax.vmap(utils.place_food_on_grid, (0, None))(state.foods, grid)
        # join all agents into 1 grid and all food into 1 grid
        agent_grid = jnp.sum(agent_grids, axis=0)
        food_grid = jnp.sum(food_grids, axis=0)

        # pad the grid so obs cannot go out of bounds
        agent_grid = jnp.pad(agent_grid, self._fov, constant_values=-1)
        food_grid = jnp.pad(food_grid, self._fov, constant_values=-1)

        # get the indexes to slice in the grid to obtain the view around the agent
        slice_len = 2 * self._fov + 1, 2 * self._fov + 1
        slice_xs, slice_ys = jax.vmap(utils.slice_around, (0, None))(
            state.agents.position, self._fov
        )

        # slice agent and food grids to obtain the view around the agent
        agents_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
            agent_grid, (slice_xs, slice_ys), slice_len
        )
        foods_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
            food_grid, (slice_xs, slice_ys), slice_len
        )
        # compute access mask (action mask in the observation); noop is always available
        access_masks = (agents_view + foods_view) == 0
        access_masks = access_masks.at[:, self._fov, self._fov].set(True)

        # compute action mask
        local_pos = jnp.array([self._fov, self._fov])
        action_mask = jax.vmap(  # vmap over all access masks
            lambda access_mask: jax.vmap(  # vmap over all moves
                lambda mv: access_mask[tuple(local_pos + mv)]
            )(MOVES)
        )(access_masks)

        return Observation(
            agents_view=jax.vmap(make_obs)(state.agents),
            action_mask=action_mask,
            step_count=state.step_count,
        )

    def observation_spec(self) -> specs.Spec[Observation]:
        max_ob = jnp.max(
            jnp.array([self._generator.max_food_level, self._generator.max_agent_level])
        )
        agents_view = specs.BoundedArray(
            shape=(self.num_agents, self.num_obs_features),
            dtype=jnp.int32,
            name="agents_view",
            minimum=-1,
            maximum=max_ob,
        )

        action_mask = specs.BoundedArray(
            shape=(self._generator.num_agents, 6),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask",
        )
        step_count = specs.BoundedArray(
            shape=(),
            dtype=jnp.int32,
            minimum=0,
            maximum=self._time_limit,
            name="step_count",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec for the Level Based Foraging environment.

        6 actions: [0,1,2,3,4,5] -> [No Op, Up, Right, Down, Left, Load].
        Since this is an environment with a multi-dimensional action space,
        it expects an array of actions of shape (num_agents,).

        Returns:
            observation_spec: `MultiDiscreteArray` of shape (num_agents,).
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.array([len(MOVES)] * self._generator.num_agents),
            dtype=jnp.int32,
            name="action",
        )

    def reward_spec(self) -> specs.Array:
        return specs.Array(shape=(self.num_agents,), dtype=float, name="reward")
