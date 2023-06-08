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

from typing import Dict, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib
from numpy.typing import NDArray

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.connector.constants import (
    AGENT_INITIAL_VALUE,
    NOOP,
    PATH,
)
from jumanji.environments.routing.connector.generator import (
    Generator,
    RandomWalkGenerator,
)
from jumanji.environments.routing.connector.reward import DenseRewardFn, RewardFn
from jumanji.environments.routing.connector.types import Agent, Observation, State
from jumanji.environments.routing.connector.utils import (
    connected_or_blocked,
    get_agent_grid,
    get_correction_mask,
    is_valid_position,
    move_agent,
    move_position,
)
from jumanji.environments.routing.connector.viewer import ConnectorViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class Connector(Environment[State]):
    """The `Connector` environment is a gridworld problem where multiple pairs of points (sets)
    must be connected without overlapping the paths taken by any other set. This is achieved
    by allowing certain points to move to an adjacent cell at each step. However, each time a
    point moves it leaves an impassable trail behind it. The goal is to connect all sets.

    - observation - `Observation`
        - action mask: jax array (bool) of shape (num_agents, 5).
        - step_count: jax array (int32) of shape ()
            the current episode step.
        - grid: jax array (int32) of shape (grid_size, grid_size)
            - with 2 agents you might have a grid like this:
              4 0 1
              5 0 1
              6 3 2
              which means agent 1 has moved from the top right of the grid down and is currently in
              the bottom right corner and is aiming to get to the middle bottom cell. Agent 2
              started in the top left and moved down once towards its target in the bottom left.

    - action: jax array (int32) of shape (num_agents,):
        - can take the values [0,1,2,3,4] which correspond to [No Op, Up, Right, Down, Left].
        - each value in the array corresponds to an agent's action.

    - reward: jax array (float) of shape ():
        - dense: reward is 1 for each successful connection on that step. Additionally,
            each pair of points that have not connected receives a penalty reward of -0.03.

    - episode termination:
        - all agents either can't move (no available actions) or have connected to their target.
        - the time limit is reached.

    - state: State:
        - key: jax PRNG key used to randomly spawn agents and targets.
        - grid: jax array (int32) of shape (grid_size, grid_size) giving the observation.
        - step_count: jax array (int32) of shape () number of steps elapsed in the current episode.

    ```python
    from jumanji.environments import Connector
    env = Connector()
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
        reward_fn: Optional[RewardFn] = None,
        time_limit: int = 50,
        viewer: Optional[Viewer[State]] = None,
    ) -> None:
        """Create the `Connector` environment.

        Args:
            generator: `Generator` whose `__call__` instantiates an environment instance.
                Implemented options are [`UniformRandomGenerator`, `RandomWalkGenerator`].
                Defaults to `RandomWalkGenerator` with `grid_size=10` and `num_agents=10`.
            reward_fn: class of type `RewardFn`, whose `__call__` is used as a reward function.
                Implemented options are [`DenseRewardFn`]. Defaults to `DenseRewardFn`.
            time_limit: the number of steps allowed before an episode terminates. Defaults to 50.
            viewer: `Viewer` used for rendering. Defaults to `ConnectorViewer` with "human" render
                mode.
        """
        self._generator = generator or RandomWalkGenerator(grid_size=10, num_agents=10)
        self._reward_fn = reward_fn or DenseRewardFn()
        self.time_limit = time_limit
        self.num_agents = self._generator.num_agents
        self.grid_size = self._generator.grid_size
        self._agent_ids = jnp.arange(self.num_agents)
        self._viewer = viewer or ConnectorViewer(
            "Connector", self.num_agents, render_mode="human"
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment.

        Args:
            key: used to randomly generate the connector grid.

        Returns:
            state: `State` object corresponding to the new state of the environment.
            timestep: `TimeStep` object corresponding to the initial environment timestep.
        """
        state = self._generator(key)

        action_mask = jax.vmap(self._get_action_mask, (0, None))(
            state.agents, state.grid
        )
        observation = Observation(
            grid=state.grid,
            action_mask=action_mask,
            step_count=state.step_count,
        )
        extras = self._get_extras(state)
        timestep = restart(observation=observation, extras=extras)
        return state, timestep

    def step(
        self, state: State, action: chex.Array
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
        new_state = State(
            grid=grid, step_count=state.step_count + 1, agents=agents, key=state.key
        )

        # Construct timestep: get reward, legal actions and done
        reward = self._reward_fn(state, action, new_state)
        action_mask = jax.vmap(self._get_action_mask, (0, None))(agents, grid)
        observation = Observation(
            grid=grid, action_mask=action_mask, step_count=new_state.step_count
        )

        done = jnp.all(jax.vmap(connected_or_blocked)(agents, action_mask))
        extras = self._get_extras(new_state)
        timestep = jax.lax.cond(
            done | (new_state.step_count >= self.time_limit),
            lambda: termination(
                reward=reward,
                observation=observation,
                extras=extras,
            ),
            lambda: transition(
                reward=reward,
                observation=observation,
                extras=extras,
            ),
        )

        return new_state, timestep

    def _step_agents(
        self, state: State, action: chex.Array
    ) -> Tuple[Agent, chex.Array]:
        """Steps all agents at the same time correcting for possible collisions.

        If a collision occurs we place the agent with the lower `agent_id` in its previous position.

        Returns:
            Tuple: (agents, grid) after having applied each agents' action
        """
        agent_ids = jnp.arange(self.num_agents)
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

    def _get_extras(self, state: State) -> Dict:
        """Computes extras metrics to be return within the timestep."""
        offset = AGENT_INITIAL_VALUE
        total_path_length = jnp.sum((offset + (state.grid - offset) % 3) == PATH)
        # Add agents' head
        total_path_length += self.num_agents
        extras = {
            "num_connections": jnp.sum(state.agents.connected),
            "ratio_connections": jnp.mean(state.agents.connected),
            "total_path_length": total_path_length,
        }
        return extras

    def render(self, state: State) -> Optional[NDArray]:
        """Render the given state of the environment.

        Args:
            state: `State` object containing the current environment state.
        """
        return self._viewer.render(state.grid)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of states.

        Args:
            states: sequence of `State` corresponding to subsequent timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            animation that can export to gif, mp4, or render with HTML.
        """
        grids = [state.grid for state in states]
        return self._viewer.animate(grids, interval, save_path)

    def close(self) -> None:
        """Perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        self._viewer.close()

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specifications of the observation of the `Connector` environment.

        Returns:
            Spec for the `Observation` whose fields are:
            - grid: BoundedArray (int32) of shape (grid_size, grid_size).
            - action_mask: BoundedArray (bool) of shape (num_agents, 5).
            - step_count: BoundedArray (int32) of shape ().
        """
        grid = specs.BoundedArray(
            shape=(self.grid_size, self.grid_size),
            dtype=jnp.int32,
            name="grid",
            minimum=0,
            maximum=self.num_agents * 3 + AGENT_INITIAL_VALUE,
        )
        action_mask = specs.BoundedArray(
            shape=(self.num_agents, 5),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask",
        )
        step_count = specs.BoundedArray(
            shape=(),
            dtype=jnp.int32,
            minimum=0,
            maximum=self.time_limit,
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
        """Returns the action spec for the Connector environment.

        5 actions: [0,1,2,3,4] -> [No Op, Up, Right, Down, Left]. Since this is an environment with
        a multi-dimensional action space, it expects an array of actions of shape (num_agents,).

        Returns:
            observation_spec: `MultiDiscreteArray` of shape (num_agents,).
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.array([5] * self.num_agents),
            dtype=jnp.int32,
            name="action",
        )
