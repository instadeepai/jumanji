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
from jumanji.environments.routing.lbf.observer import GridObserver, LbfObserver
from jumanji.environments.routing.lbf.types import Food, Observation, State
from jumanji.types import TimeStep, restart, termination, transition


class LevelBasedForaging(Environment[State]):
    """
    An implementation of the Level Based Foraging environment where agents need
    to work cooperatively to collect food.

    See original implementation: https://github.com/semitable/lb-foraging/tree/master

    - observation: `Observation`
        - agent_views: this depends on the `observer` passed to `__init__`. It can either be a
            `GridObserver` or a `VectorObserver`.
            The `GridObserver` returns an agent's view with a shape of (num_agents, 3, 2 * fov + 1, 2 * fov +1).
            The `VectorObserver` returns an agent's view with a shape of (num_agents, 3 * num_foods + 3 * num_agents).
            See the docs of those classes for more details.
        - action_mask: jax array (bool) of shape (num_agents, 6)
            indicates for each agent which of the size actions (no-op, up, right, down, left, load) is allowed.
        - step_count: (int32)
            the number of step since the beginning of the episode.

    - action: jax array (int32) of shape (num_agents,)
        the action for each agent: (0: noop, 1: up, 2: right, 3: down, 4: left, 5: load).

    - reward: jax array (float) of shape (num_agents,)
        When one or more agents load a food, the food level is rewarded to the agents weighted by the level
        of each agent. Then the reward is normalised so that at the end, the sum of the rewards
        (if all foods have been picked-up) is one.

    - episode termination:
        - All foods have been eaten.
        - The number of steps is greater than the limit.

    - state: `State`
        - agents: stacked pytree of `Agent` objects of length `num_agents`.
            - Agent:
                - id: jax array (int32) of shape ().
                - position: jax array (int32) of shape (2,).
                - level: jax array (int32) of shape ().
                - loading: jax array (bool) of shape ().
        - foods: stacked pytree of `Food` objects of length `num_food`.
            - Food:
                - id: jax array (int32) of shape ().
                - position: jax array (int32) of shape (2,).
                - level: jax array (int32) of shape ().
                - eaten: jax array (bool) of shape ().
        - step_count: jax array (int32) of shape ()
            the number of steps since the beginning of the episode.
        - key: jax array (uint) of shape (2,)
            jax random generation key. Ignored since the environment is deterministic.

    ```python
    from jumanji.environments import LevelBasedForaging
    env = LevelBasedForaging()
    key = jax.random.key(0)
    state, timestep = jax.jit(env.reset)(key)
    env.render(state)
    action = env.action_spec().generate_value()
    state, timestep = jax.jit(env.step)(state, action)
    env.render(state)
    ```
    """

    def __init__(
        self,
        generator: Optional[Generator] = None,
        observer: Optional[LbfObserver] = None,
        time_limit: int = 50,
    ) -> None:
        """
        Instantiates a `LevelBasedForaging` environment.

        Defaults are equivalent to `Foraging-10x10-3p-3f-v2` in the original implementation.
        https://github.com/semitable/lb-foraging/tree/master

        Args:
            generator: a `Generator` object that generates the initial state of the environment.
                Defaults to a `RandomGenerator` with the following parameters:
                    - grid_size: 10
                    - num_agents: 3
                    - num_food: 3
                    - max_agent_level: 2
                    - max_food_level: 6
            observer: an `Observer` object that generates the observation of the environment.
                Either a `GridObserver` or a `VectorObserver`.
                Defaults to a `GridObserver` with a field of view of 10.
            time_limit: the maximum number of steps in an episode. Defaults to 50.
        """
        super().__init__()

        self._generator = generator or RandomGenerator(
            grid_size=10, num_agents=3, num_food=3, max_agent_level=2, max_food_level=6
        )
        self._observer = observer or GridObserver(
            fov=10, grid_size=self._generator.grid_size
        )
        self._time_limit = time_limit

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Resets the environment.

        Args:
            key: used to randomly generate the new `State`.

        Returns:
            state: `State` object corresponding to the new state of the environment.
            timestep: `TimeStep` object corresponding to the initial environment timestep.
        """
        state = self._generator(key)
        observation = self._observer.state_to_observation(state)

        return state, restart(observation, shape=self._generator.num_agents)

    def step(self, state: State, actions: chex.Array) -> Tuple[State, TimeStep]:
        """Simulate one step of the environment.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the actions to take for each agent.
                - 0 no op
                - 1 move up
                - 2 move right
                - 3 move down
                - 4 move left
                - 5 load

        Returns:
            state: `State` object corresponding to the next state of the environment.
            timestep: `TimeStep` object corresponding the timestep returned by the environment.
        """
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
        """Returns a reward for all agents given all foods.

        Args:
            foods: all the foods in the environment.
            adj_agent_levels: the level of all agents adjacent to all foods.
                Shape (num_foods, num_agents).
            eaten: whether the food was eaten or not (this step).
        """
        # Get reward per food for all foods and agents (by vmapping over foods).
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
            adj_agent_levels: the level of the agents adjacent to this food,
                this is 0 if the agent is not adjacent.
                Shape - (num_agents,).
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
        """Specifications of the observation of the `LevelBasedForaging` environment.

        The spec's shape depends on the `observer` passed to `__init__`.

        The GridObserver returns an agent's view with a shape of (num_agents, 3, 2 * fov + 1, 2 * fov +1).
        The VectorObserver returns an agent's view with a shape of (num_agents, 3 * num_foods + 3 * num_agents).
        See a more detailed description of the observations in the docs of `GridObserver` and `VectorObserver`.

        Returns:
            Spec for the `Observation` whose fields are:
            - grid: BoundedArray (int32) - shape is dependent on observer, described above.
            - action_mask: BoundedArray (bool) of shape (num_agents, 6).
            - step_count: BoundedArray (int32) of shape ().
        """
        return self._observer.observation_spec(
            self._generator.num_agents,
            self._generator._num_food,
            self._generator._max_agent_level,
            self._generator._max_food_level,
            self._time_limit,
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec for the Level Based Foraging environment.

        6 actions: [0,1,2,3,4,5] -> [No Op, Up, Right, Down, Left, Load].
        Since this is an environment with a multi-dimensional action space,
        it expects an array of actions of shape (num_agents,).

        Returns:
            action_spec: `specs.MultiDiscreteArray` of shape (num_agents,).
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.array([len(MOVES)] * self._generator.num_agents),
            dtype=jnp.int32,
            name="action",
        )

    def reward_spec(self) -> specs.Array:
        """Returns the reward specification for the `LevelBasedForaging` environment.

        Since this is a multi-agent environment each agent gets its own reward.

        Returns:
            reward_spec: `specs.Array` of shape (num_agents,)
        """
        return specs.Array(
            shape=(self._generator.num_agents,), dtype=float, name="reward"
        )

    def discount_spec(self) -> specs.BoundedArray:
        """Returns the discount specification for the `LevelBasedForaging` environment.

        Since this is a multi-agent environment each agent gets its own discount.

        Returns:
            discount_spec: `specs.BoundedArray` of shape (num_agents,) with values in [0, 1].
        """
        return specs.BoundedArray(
            shape=(self._generator.num_agents,),
            dtype=float,
            minimum=0.0,
            maximum=1.0,
            name="discount",
        )
