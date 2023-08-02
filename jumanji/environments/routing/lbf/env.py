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
        fov: int = 1,
        time_limit=500,
    ) -> None:
        super().__init__()

        self._generator = generator or UniformRandomGenerator(
            grid_size=5, num_agents=4, num_food=5, max_agent_level=3, max_food_level=3
        )
        self._fov = fov
        self._time_limit = time_limit

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        state = self._generator(key)
        timestep = self._state_to_timestep(state, jnp.zeros(self._generator.num_agents))

        return state, timestep

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

        return state, self._state_to_timestep(state, reward)

    # Do we make this a param of the env even though there's probably only a single type of reward?
    def get_reward(
        self, foods: Food, adj_agent_levels: chex.Array, eaten: chex.Array
    ) -> chex.Array:
        """Returns a reward for all agents given all foods."""
        # Get reward per food for all food (by vmapping over foods).
        # Then sum that reward on agent dim to get reward per agent.
        return jnp.sum(
            jax.vmap(self._reward_per_food)(foods, adj_agent_levels, eaten),
            axis=0,
        )

    def _reward_per_food(
        self, food: Food, adj_agent_levels: chex.Array, eaten: chex.Array
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

        def _reward(adj_level_if_eaten: chex.Numeric, food: Food) -> chex.Array:
            """Returns the reward for a single agent given it's level if it was adjacent."""
            reward = adj_level_if_eaten * food.level
            normalizer = total_adj_level * self._generator.num_food
            return reward / normalizer

        return jax.vmap(_reward, (0, None))(adj_levels_if_eaten, food)

    def _state_to_timestep(self, state: State, reward: chex.Array) -> TimeStep:
        grid = jnp.zeros((self._generator.grid_size, self._generator.grid_size))
        agent_grid = jax.vmap(utils.place_agent_on_grid, (0, None))(state.agents, grid)
        food_grid = jax.vmap(utils.place_food_on_grid, (0, None))(state.foods, grid)

        grids, action_masks = jax.vmap(self._get_agent_obs, (0, None, None))(
            state.agents, agent_grid, food_grid
        )
        observation = Observation(
            agent_views=grids,
            action_mask=action_masks,
            step_count=state.step_count,
        )

        # First condition is truncation, second is termination. Jumanji doesn't support truncation yet.
        done = state.step_count >= self._time_limit | jnp.all(state.foods.eaten)

        return jax.lax.cond(done, termination, transition, observation, reward)

    # This is the new observation that lbf offers.
    # The old obs was used in the paper, but these obs make more sense and are implemented
    # in the original repo.
    def _get_agent_obs(
        self, agent: Agent, agent_grid: chex.Array, food_grid: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        slice_coords = utils.slice_around(agent.position, self._fov)

        agent_view = jnp.pad(agent_grid, self._fov, constant_values=-1)[slice_coords]
        food_view = jnp.pad(food_grid, self._fov, constant_values=-1)[slice_coords]
        access_mask = (agent_view + food_view) == 0
        # noop is always available
        access_mask = access_mask.at[self._fov, self._fov].set(True)

        # todo: should this be it's own function?
        local_pos = jnp.array([self._fov, self._fov])
        action_mask = jax.vmap(lambda mv: access_mask[tuple(local_pos + mv)])(MOVES)

        return jnp.stack([agent_view, food_view, access_mask]), action_mask

    def observation_spec(self) -> specs.Spec[Observation]:
        grid = specs.BoundedArray(
            shape=(self._generator.grid_size, self._generator.grid_size),
            dtype=jnp.int32,
            name="grid",
            minimum=0,
            maximum=jnp.max(
                self._generator.max_food_level, self._generator.max_agent_level
            ),
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
            grid=grid,
            action_mask=action_mask,
            step_count=step_count,
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec for the Level Based Foraging environment.

        6 actions: [0,1,2,3,4,5] -> [No Op, Up, Right, Down, Left, Load]. Since this is an environment with
        a multi-dimensional action space, it expects an array of actions of shape (num_agents,).

        Returns:
            observation_spec: `MultiDiscreteArray` of shape (num_agents,).
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.array([6] * self._generator.num_agents),
            dtype=jnp.int32,
            name="action",
        )
