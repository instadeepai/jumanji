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

from typing import Dict, Optional, Sequence, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import matplotlib
from numpy.typing import NDArray

import jumanji.environments.routing.lbf.utils as utils
from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.lbf.constants import MOVES
from jumanji.environments.routing.lbf.generator import RandomGenerator
from jumanji.environments.routing.lbf.observer import GridObserver, VectorObserver
from jumanji.environments.routing.lbf.types import Food, Observation, State
from jumanji.environments.routing.lbf.viewer import LevelBasedForagingViewer
from jumanji.types import TimeStep, restart, termination, transition, truncation
from jumanji.viewer import Viewer


class LevelBasedForaging(Environment[State]):
    """
    An implementation of the Level-Based Foraging environment where agents need to
    cooperate to collect food and split the reward.

    Original implementation: https://github.com/semitable/lb-foraging/tree/master

    - `observation`: `Observation`
        - `agent_views`: Depending on the `observer` passed to `__init__`, it can be a
          `GridObserver` or a `VectorObserver`.
            - `GridObserver`: Returns an agent's view with a shape of
              (num_agents, 3, 2 * fov + 1, 2 * fov +1).
            - `VectorObserver`: Returns an agent's view with a shape of
              (num_agents, 3 * (num_food + num_agents).
        - `action_mask`: JAX array (bool) of shape (num_agents, 6)
          indicating for each agent which size actions
          (no-op, up, down, left, right, load) are allowed.
        - `step_count`: int32, the number of steps since the beginning of the episode.

    - `action`: JAX array (int32) of shape (num_agents,). The valid actions for each
        agent are (0: noop, 1: up, 2: down, 3: left, 4: right, 5: load).

    - `reward`: JAX array (float) of shape (num_agents,)
        When one or more agents load food, the food level is rewarded to the agents, weighted
        by the level of each agent. The reward is then normalized so that, at the end,
        the sum of the rewards (if all food items have been picked up) is one.

    - Episode Termination:
        - All food items have been eaten.
        - The number of steps is greater than the limit.

    - `state`: `State`
        - `agents`: Stacked Pytree of `Agent` objects of length `num_agents`.
            - `Agent`:
                - `id`: JAX array (int32) of shape ().
                - `position`: JAX array (int32) of shape (2,).
                - `level`: JAX array (int32) of shape ().
                - `loading`: JAX array (bool) of shape ().
        - `food_items`: Stacked Pytree of `Food` objects of length `num_food`.
            - `Food`:
                - `id`: JAX array (int32) of shape ().
                - `position`: JAX array (int32) of shape (2,).
                - `level`: JAX array (int32) of shape ().
                - `eaten`: JAX array (bool) of shape ().
        - `step_count`: JAX array (int32) of shape (), the number of steps since the beginning
          of the episode.
        - `key`: JAX array (uint) of shape (2,)
            JAX random generation key. Ignored since the environment is deterministic.

    Example:
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

    Initialization Args:
    - `generator`: A `Generator` object that generates the initial state of the environment.
        Defaults to a `RandomGenerator` with the following parameters:
            - `grid_size`: 8
            - `fov`: 8 (full observation of the grid)
            - `num_agents`: 2
            - `num_food`: 2
            - `max_agent_level`: 2
            - `force_coop`: True
    - `time_limit`: The maximum number of steps in an episode. Defaults to 200.
    - `grid_observation`: If `True`, the observer generates a grid observation (default is `False`).
    - `normalize_reward`: If `True`, normalizes the reward (default is `True`).
    - `penalty`: The penalty value (default is 0.0).
    - `viewer`: Viewer to render the environment. Defaults to `LevelBasedForagingViewer`.
    """

    def __init__(
        self,
        generator: Optional[RandomGenerator] = None,
        viewer: Optional[Viewer[State]] = None,
        time_limit: int = 200,
        grid_observation: bool = False,
        normalize_reward: bool = True,
        penalty: float = 0.0,
    ) -> None:
        super().__init__()

        self._generator = generator or RandomGenerator(
            grid_size=8,
            fov=8,
            num_agents=2,
            num_food=2,
            force_coop=True,
        )
        self._time_limit = time_limit
        self._grid_size: int = self._generator.grid_size
        self._num_agents: int = self._generator.num_agents
        self._num_food: int = self._generator.num_food
        self._normalize_reward = normalize_reward
        self._penalty = penalty
        self.num_obs_features = utils.calculate_num_observation_features(
            self._num_food, self._num_agents
        )
        self._observer: Union[VectorObserver, GridObserver]
        if not grid_observation:
            self._observer = VectorObserver(
                fov=self._generator.fov,
                grid_size=self._grid_size,
                num_agents=self._num_agents,
                num_food=self._num_food,
            )

        else:
            self._observer = GridObserver(
                fov=self._generator.fov,
                grid_size=self._grid_size,
                num_agents=self._num_agents,
                num_food=self._num_food,
            )

        # create viewer for rendering environment
        self._viewer = viewer or LevelBasedForagingViewer(
            self._grid_size, "LevelBasedForaging"
        )

    @property
    def time_limit(self) -> int:
        return self._time_limit

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def fov(self) -> int:
        return self._generator.fov

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def num_food(self) -> int:
        return self._num_food

    @property
    def max_agent_level(self) -> int:
        return self._generator.max_agent_level

    def __repr__(self) -> str:
        return (
            "LevelBasedForaging(\n"
            + f"\t grid_width={self._grid_size},\n"
            + f"\t grid_height={self._grid_size},\n"
            + f"\t num_agents={self._num_agents}, \n"
            + f"\t num_food={self._num_food}, \n"
            + f"\t max_agent_level={self._generator.max_agent_level}\n"
            ")"
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Resets the environment.

        Args:
            key (chex.PRNGKey): Used to randomly generate the new `State`.

        Returns:
            Tuple[State, TimeStep]: `State` object corresponding to the new initial state
            of the environment and `TimeStep` object corresponding to the initial timestep.
        """
        state = self._generator(key)
        observation = self._observer.state_to_observation(state)
        timestep = restart(observation, shape=self._num_agents)
        timestep.extras = self._get_extra_info(state, timestep)

        return state, timestep

    def step(self, state: State, actions: chex.Array) -> Tuple[State, TimeStep]:
        """Simulate one step of the environment.

        Args:
            state (State): State  containing the dynamics of the environment.
            actions (chex.Array): Array containing the actions to take for each agent.

        Returns:
            Tuple[State, TimeStep]: `State` object corresponding to the next state and
            `TimeStep` object corresponding the timestep returned by the environment.
        """
        # Move agents, fix collisions that may happen and set loading status.
        moved_agents = utils.update_agent_positions(
            state.agents, actions, state.food_items, self._grid_size
        )

        # Eat the food
        food_items, eaten_this_step, adj_loading_agents_levels = jax.vmap(
            utils.eat_food, (None, 0)
        )(moved_agents, state.food_items)

        reward = self.get_reward(food_items, adj_loading_agents_levels, eaten_this_step)

        state = State(
            agents=moved_agents,
            food_items=food_items,
            step_count=state.step_count + 1,
            key=state.key,
        )
        observation = self._observer.state_to_observation(state)

        # First condition is truncation, second is termination.
        terminate = jnp.all(state.food_items.eaten)
        truncate = state.step_count >= self.time_limit

        timestep = jax.lax.switch(
            terminate + 2 * truncate,
            [
                # !terminate !trunc
                lambda rew, obs: transition(
                    reward=rew, observation=obs, shape=self._num_agents
                ),
                # terminate !truncate
                lambda rew, obs: termination(
                    reward=rew, observation=obs, shape=self._num_agents
                ),
                # !terminate truncate
                lambda rew, obs: truncation(
                    reward=rew, observation=obs, shape=self._num_agents
                ),
                # terminate truncate
                lambda rew, obs: termination(
                    reward=rew, observation=obs, shape=self._num_agents
                ),
            ],
            reward,
            observation,
        )
        timestep.extras = self._get_extra_info(state, timestep)

        return state, timestep

    def _get_extra_info(self, state: State, timestep: TimeStep) -> Dict:
        """Computes extras metrics to be returned within the timestep."""
        n_eaten = state.food_items.eaten.sum() + timestep.extras.get(
            "eaten_food", jnp.float32(0)
        )
        percent_eaten = (n_eaten / self.num_food) * 100
        return {"num_eaten": n_eaten, "percent_eaten": percent_eaten}

    def get_reward(
        self,
        food_items: Food,
        adj_loading_agents_levels: chex.Array,
        eaten_this_step: chex.Array,
    ) -> chex.Array:
        """Returns a reward for all agents given all food items.

        Args:
            food_items (Food): All the food items in the environment.
            adj_loading_agents_levels (chex.Array): The level of all agents adjacent to all foods.
            eaten_this_step (chex.Array): Whether the food was eaten or not (this step).
        """

        def get_reward_per_food(
            food: Food,
            adj_loading_agents_levels: chex.Array,
            eaten_this_step: chex.Array,
        ) -> chex.Array:
            """Returns the reward for all agents given a single food."""

            # If the food has already been eaten or is not loaded, the sum will be equal to 0
            sum_agents_levels = jnp.sum(adj_loading_agents_levels)

            # Penalize agents for not being able to cooperate and eat food
            penalty = jnp.where(
                (sum_agents_levels != 0) & (sum_agents_levels < food.level),
                self._penalty,
                0,
            )

            # Zero out all agents if food was not eaten and add penalty
            reward = (
                adj_loading_agents_levels * eaten_this_step * food.level
            ) - penalty

            # jnp.nan_to_num: Used in the case where no agents are adjacent to the food
            normalizer = sum_agents_levels * total_food_level
            reward = jnp.where(
                self._normalize_reward, jnp.nan_to_num(reward / normalizer), reward
            )

            return reward

        # Get reward per food for all food items,
        # then sum it on the agent dimension to get reward per agent.
        total_food_level = jnp.sum(food_items.level)
        reward_per_food = jax.vmap(get_reward_per_food, in_axes=(0, 0, 0))(
            food_items, adj_loading_agents_levels, eaten_this_step
        )
        return jnp.sum(reward_per_food, axis=0)

    def render(self, state: State) -> Optional[NDArray]:
        """Renders the current state of the `LevelBasedForaging` environment.

        Args:
            state (State): The current environment state to be rendered.

        Returns:
            Optional[NDArray]: Rendered environment state.
        """
        return self._viewer.render(state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Creates an animation from a sequence of states.

        Args:
            states (Sequence[State]): Sequence of `State` corresponding to subsequent timesteps.
            interval (int): Delay between frames in milliseconds, default to 200.
            save_path (Optional[str]): The path where the animation file should be saved.

        Returns:
            matplotlib.animation.FuncAnimation: Animation object that can be saved as a GIF, MP4,
            or rendered with HTML.
        """
        return self._viewer.animate(
            states=states, interval=interval, save_path=save_path
        )

    def close(self) -> None:
        """Perform any necessary cleanup."""
        self._viewer.close()

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specifications of the observation of the environment.

        The spec's shape depends on the `observer` passed to `__init__`.

        The GridObserver returns an agent's view with a shape of
            (num_agents, 3, 2 * fov + 1, 2 * fov +1).
        The VectorObserver returns an agent's view with a shape of
        (num_agents, 3 * num_food + 3 * num_agents).
        See a more detailed description of the observations in the docs
        of `GridObserver` and `VectorObserver`.

        Returns:
            specs.Spec[Observation]: Spec for the `Observation` with fields grid,
            action_mask, and step_count.
        """
        max_food_level = self._num_agents * self._generator.max_agent_level
        return self._observer.observation_spec(
            self._generator.max_agent_level,
            max_food_level,
            self.time_limit,
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec for the Level Based Foraging environment.

        Returns:
            specs.MultiDiscreteArray: Action spec for the environment with shape (num_agents,).
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.array([len(MOVES)] * self._num_agents),
            dtype=jnp.int32,
            name="action",
        )

    def reward_spec(self) -> specs.Array:
        """Returns the reward specification for the `LevelBasedForaging` environment.

        Since this is a multi-agent environment each agent gets its own reward.

        Returns:
            specs.Array: Reward specification, of shape (num_agents,) for the  environment.
        """
        return specs.Array(shape=(self._num_agents,), dtype=float, name="reward")

    def discount_spec(self) -> specs.BoundedArray:
        """Describes the discount returned by the environment.

        Returns:
            discount_spec: a `specs.BoundedArray` spec.
        """
        return specs.BoundedArray(
            shape=(self.num_agents,),
            dtype=float,
            minimum=0.0,
            maximum=1.0,
            name="discount",
        )
