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

from typing import Any, Tuple, Type

import chex
import jax
import jax.numpy as jnp
from chex import PRNGKey

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.connector.constants import AGENT_INITIAL_VALUE, NOOP
from jumanji.environments.routing.connector.generator import (
    InstanceGenerator,
    UniformRandomGenerator,
)
from jumanji.environments.routing.connector.reward import RewardFn, SparseRewardFn
from jumanji.environments.routing.connector.types import Agent, Observation, State
from jumanji.environments.routing.connector.utils import (
    connected_or_blocked,
    get_agent_grid,
    get_correction_mask,
    is_valid_position,
    move_agent,
    move_position,
    switch_perspective,
)
from jumanji.types import Action, TimeStep, restart, termination, transition


class Connector(Environment[State]):
    """The Connector environment is a multi-agent gridworld problem where each agent must connect a
    start to a target. However when moving through this gridworld the agent leaves an impassable
    trail behind it. Therefore agents must connect to their targets without overlapping the routes
    taken by any other agent.

    - observation - Observation
        - action mask: jax array (bool) of shape (num_agents, 5).
        - step: (int) the current episode step.
        - grid: jax array (int) of shape (num_agents, size, size):
            - each 2d array (size, size) along axis 0 is the agent's local observation.
            - agents have ids from 0 to (num_agents - 1)
            - with 2 agents you might have a grid like this:
              4 0 1
              5 0 1
              6 3 2
              which means agent 1 has moved from the top right of the grid down and is currently in
              the bottom right corner and is aiming to get to the middle bottom cell. Agent 2
              started in the top left and moved down once towards its target in the bottom left.

              This would just be agent 0's view, the numbers would be flipped for agent 1's view.
              So the full observation would be of shape (2, 3, 3).

    - action: jax array (int) of shape (num_agents,):
        - can take the values [0,1,2,3,4] which correspond to [No Op, Up, Right, Down, Left].
        - each value in the array corresponds to an agent's action.

    - reward_fn: function that takes old state, new state and action returns a reward for each
        agent.

    - episode termination: if an agent can't move, or the horizon is reached, or the agent connects
        to its target, it is considered done. Once all agents are done, the episode will terminate.
        - timestep discounts are of shape (num_agents,)

    - state: State:
        - key: jax PRNG key used to randomly spawn agents and targets.
        - grid: jax array (int) of shape (size, size) which corresponds to agent 0's observation.
        - step: (int) number of steps passed in the current episode.
    """

    def __init__(
        self,
        size: int = 8,
        num_agents: int = 3,
        reward_fn: Type[RewardFn] = SparseRewardFn,
        horizon: int = 50,
        generator: Type[InstanceGenerator] = UniformRandomGenerator,
        **reward_fn_kwargs: Any
    ) -> None:
        """Create the Connector Environment.

        Args:
            size: number of rows and columns in the grid.
            num_agents: number of agents in the grid (or equivalently, the number of targets).
            reward_fn: class of type RewardFn, whose __call__ is used as a reward function.
            horizon: the number of steps allowed before an episode terminates.
            generator: an InstanceGenerator that generates new grids.
            reward_fn_kwargs: arguments passed to the reward_fn's __init__ to configure it.
        """
        super().__init__()

        self._size = size
        self._num_agents = num_agents
        self._horizon = horizon

        self.agent_ids = jnp.arange(self._num_agents)

        self._reward_fn = reward_fn(**reward_fn_kwargs)
        self._generator = generator(size, num_agents)

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment.

        Args:
            key: used to randomly generate the connector grid.

        Returns:
            state: `State` object corresponding to the new state of the environment.
            timestep: `TimeStep` object corresponding to the initial environment timestep.
        """
        generator_key, key = jax.random.split(key)
        state = self._generator(generator_key)

        action_mask = jax.vmap(self._get_action_mask, (0, None))(
            state.agents, state.grid
        )
        observation = Observation(
            grid=self._obs_from_grid(state.grid),
            action_mask=action_mask,
            step=state.step,
        )
        timestep = restart(observation=observation, shape=(self._num_agents,))

        return (
            State(key=key, grid=state.grid, step=jnp.int32(0), agents=state.agents),
            timestep,
        )

    def step(
        self,
        state: State,
        action: Action,
    ) -> Tuple[State, TimeStep[Observation]]:
        """Perform an environment step.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the actions to take for each agent.
                - 0 no op
                - 1 move up
                - 2 move right
                - 3 move down
                - 4 move left

        Returns:
            state: `State` object corresponding to the next state of the environment.
            timestep: `TimeStep` object corresponding the timestep returned by the environment.
        """
        agents, grid = self._step_agents(state, action)
        new_state = State(key=state.key, grid=grid, step=state.step + 1, agents=agents)

        # Construct timestep: get observations, rewards, discounts
        grids = self._obs_from_grid(grid)
        reward = self._reward_fn(state, new_state, action)
        action_mask = jax.vmap(self._get_action_mask, (0, None))(agents, grid)
        observation = Observation(
            grid=grids, action_mask=action_mask, step=new_state.step
        )

        dones = jax.vmap(connected_or_blocked)(agents, action_mask)
        discount = jnp.asarray(jnp.logical_not(dones), dtype=float)

        timestep = jax.lax.cond(
            dones.all() | (new_state.step >= self._horizon),
            lambda: termination(
                reward=reward, observation=observation, shape=self._num_agents
            ),
            lambda: transition(
                reward=reward,
                observation=observation,
                discount=discount,
                shape=self._num_agents,
            ),
        )

        return new_state, timestep

    def _step_agents(self, state: State, action: Action) -> Tuple[Agent, chex.Array]:
        """Steps all agents at the same time correcting for possible collisions.

        If a collision occurs we place the agent with the lower `agent_id` in its previous position.

        Returns:
            Tuple: (agents, grid) after having applied each agents action
        """
        agent_ids = jnp.arange(self._num_agents)
        # Step all agents at the same time (separately) and return all of the grids
        agents, grids = jax.vmap(self._step_agent, in_axes=(0, None, 0))(
            state.agents, state.grid, action
        )

        # Get grids with only values related to a single agent.
        # For example: remove all other agents from agent 1's grid. Do this for all agents.
        agent_grids = jax.vmap(get_agent_grid)(agent_ids, grids)
        joined_grid = jnp.max(agent_grids, 0)  # join the grids

        # Create a correction mask for possible collisions (see the docs of `get_correction_mask`)
        correction_fn = jax.vmap(get_correction_mask, in_axes=(None, None, 0))
        correction_masks, collided_agents = correction_fn(
            state.grid, joined_grid, agent_ids
        )
        correction_mask = jnp.sum(correction_masks, 0)

        # Correct state.agents
        # Get the correct agents, either old agents (if collision) or new agents if no collision
        agents = jax.vmap(
            lambda collided, old_agent, new_agent: jax.lax.cond(
                collided,
                lambda: old_agent,
                lambda: new_agent,
            )
        )(collided_agents, state.agents, agents)
        # Create the new grid by fixing old one with correction mask and adding the obstacles
        return agents, joined_grid + correction_mask

    def _step_agent(
        self, agent: Agent, grid: chex.Array, action: chex.Numeric
    ) -> Tuple[Agent, chex.Array]:
        """Moves the agent according to the given action if it is possible.

        Returns:
            Tuple: (agent, grid) after having applied the given action.
        """
        new_pos = move_position(agent.position, action)

        new_agent, new_grid = jax.lax.cond(
            is_valid_position(grid, agent, new_pos) & (action != NOOP),
            move_agent,
            lambda *_: (agent, grid),
            agent,
            grid,
            new_pos,
        )

        return new_agent, new_grid

    def _obs_from_grid(self, grid: chex.Array) -> chex.Array:
        """Gets the observation vector for all agents."""
        return jax.vmap(switch_perspective, (None, 0, None))(
            grid, self.agent_ids, self._num_agents
        )

    def _get_action_mask(self, agent: Agent, grid: chex.Array) -> chex.Array:
        """Gets an agent's action mask."""
        # Don't check action 0 because no-op is always valid
        actions = jnp.arange(1, 5)

        def is_valid_action(action: int) -> chex.Array:
            agent_pos = move_position(agent.position, action)
            return is_valid_position(grid, agent, agent_pos)

        mask = jnp.ones(5, dtype=bool)
        mask = mask.at[actions].set(jax.vmap(is_valid_action)(actions))
        return mask

    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec for Connector environment.

        This observation contains the grid for each agent, the action mask for each agent and
        the current step.

        Returns:
            observation_spec: an `ObservationSpec` which contains the grid and the action mask spec.
        """
        grid = specs.BoundedArray(
            shape=(self._num_agents, self._size, self._size),
            dtype=int,
            name="observation",
            minimum=0,
            maximum=self._num_agents * 3 + AGENT_INITIAL_VALUE,
        )
        action_mask = specs.BoundedArray(
            shape=(self._num_agents, 5),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask",
        )
        step = specs.BoundedArray(
            shape=(), dtype=int, minimum=0, maximum=self._horizon, name="step"
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            grid=grid,
            action_mask=action_mask,
            step=step,
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec for the Connector environment.

        5 actions: [0,1,2,3,4] -> [No Op, Up, Right, Down, Left]. Since this is a multi-agent
        environment, the environment expects an array of actions of shape (num_agents,).

        Returns:
            observation_spec: MultiDiscreteArray of shape (num_agents,).
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.array([5] * self._num_agents),
            name="action",
        )

    def reward_spec(self) -> specs.Array:
        """
        Returns:
            reward_spec: a `specs.Array` spec of shape (num_agents,). One for each agent.
        """
        return specs.Array(shape=(self._num_agents,), dtype=float, name="reward")

    def discount_spec(self) -> specs.BoundedArray:
        """
        Returns:
            discount_spec: a `specs.Array` spec of shape (num_agents,). One for each agent
        """
        return specs.BoundedArray(
            shape=(self._num_agents,),
            dtype=float,
            minimum=0.0,
            maximum=1.0,
            name="discount",
        )
