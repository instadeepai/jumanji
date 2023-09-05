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
from jumanji.environments.routing.lbf.generator import Generator, RandomGenerator
from jumanji.environments.routing.lbf.observer import LbfGridObserver, LbfObserver
from jumanji.environments.routing.lbf.types import Food, Observation, State
from jumanji.types import TimeStep, restart, termination, transition


class LevelBasedForaging(Environment[State]):
    def __init__(
        self,
        generator: Optional[Generator] = None,
        observer: Optional[LbfObserver] = None,
        fov: int = 10,
        time_limit: int = 50,
    ) -> None:
        super().__init__()

        self._generator = generator or RandomGenerator(
            grid_size=10, num_agents=3, num_food=3, max_agent_level=2, max_food_level=6
        )
        self._observer = observer or LbfGridObserver(
            fov=fov, grid_size=self._generator.grid_size
        )
        self._fov = fov
        self._time_limit = time_limit
        self.time_limit = time_limit
        self.num_agents = self._generator.num_agents
        self.num_obs_features = self.num_agents * 3 + self._generator.num_food * 3

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        state = self._generator(key)
        observation = self._observer.state_to_observation(state)

        return state, restart(observation, shape=self._generator.num_agents)

    def step(self, state: State, actions: chex.Array) -> Tuple[State, TimeStep]:
        # Move agents, fix collisions that may happen and set loading status.
        moved_agents = jax.vmap(utils.move, (0, 0, None, None, None))(
            state.agents,
            actions,
            state.foods,
            state.agents,
            self._generator.grid_size,
        )
        # check that no two agent share the same position after moving
        moved_agents = utils.fix_collisions(moved_agents, state.agents)

        # set agent's loading status
        moved_agents = jax.vmap(
            lambda agent, action: agent.replace(loading=action == LOAD)
        )(moved_agents, actions)

        # eat food
        foods, eaten_this_step, adj_loading_level = jax.vmap(utils.eat, (None, 0))(
            moved_agents, state.foods
        )

        reward = self.get_reward(foods, adj_loading_level, eaten_this_step)

        state = State(
            agents=moved_agents,
            foods=foods,
            step_count=state.step_count + 1,
            key=state.key,
        )

        observation = self._observer.state_to_observation(state)
        # First condition is truncation, second is termination.
        done = (state.step_count >= self._time_limit) | jnp.all(state.foods.eaten)
        timestep = jax.lax.cond(
            done,
            lambda: termination(reward, observation, shape=self._generator.num_agents),
            lambda: transition(reward, observation, shape=self._generator.num_agents),
        )

        return state, timestep

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
            eaten: whether the food was eaten or not (this step).
            total_food_level: the sum of all food levels in the environment.
        """
        # zero out all agents if food was not eaten
        adj_levels_if_eaten = adj_agent_levels * eaten

        reward = adj_levels_if_eaten * food.level
        normalizer = jnp.sum(adj_agent_levels) * total_food_level
        # It's often the case that no agents are adjacent to the food
        # so we need to avoid dividing by 0 -> nan_to_num
        return jnp.nan_to_num(reward / normalizer)

    def observation_spec(self) -> specs.Spec[Observation]:
        return self._observer.observation_spec(
            self._generator.num_agents,
            self._generator.num_food,
            self._generator.max_agent_level,
            self._generator.max_food_level,
            self.time_limit,
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
