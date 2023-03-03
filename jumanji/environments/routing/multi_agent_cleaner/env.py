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

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.multi_agent_cleaner.constants import (
    CLEAN,
    DIRTY,
    MOVES,
    WALL,
)
from jumanji.environments.routing.multi_agent_cleaner.env_viewer import CleanerViewer
from jumanji.environments.routing.multi_agent_cleaner.instance_generator import (
    Generator,
    RandomGenerator,
)
from jumanji.environments.routing.multi_agent_cleaner.types import Observation, State
from jumanji.types import Action, TimeStep, restart, termination, transition


class Cleaner(Environment[State]):
    """A JAX implementation of the 'Multi-Agent Cleaner' game.

    - observation: Observation
        - grid: jax array (int) containing the state of the board:
            0 for dirty tile, 1 for clean tile, 2 for wall.
        - agents_locations: jax array (int) of size (num_agents, 2) containing
            the location of each agent on the board.
        - action_mask: jax array (bool) of size (num_agents, 4) stating for each agent
            if each of the four actions (up, right, down, left) is allowed.

    - action: jax array (int) of shape (num_agents,) containing the action for each agent.
        (0: up, 1: right, 2: down, 3: left)

    - reward: global reward, +1 every time a tile is cleaned.

    - episode termination:
        - All tiles are clean.
        - The number of steps is greater than the limit.
        - An invalid action is selected for any of the agents.

    - state: State
        - grid: jax array (int) containing the state of the board:
            0 for dirty tile, 1 for clean tile, 2 for a wall.
        - agents_locations: jax array (int) of size (num_agents, 2) containing
            the location of each agent on the board.
        - action_mask: jax array (bool) of size (num_agents, 4) stating for each agent
            if each of the four actions (up, right, down, left) is allowed.
        - step_count: the number of steps from the beginning of the environment.
        - key: jax random generation key. Ignored since the environment is deterministic.
    """

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        num_agents: int,
        step_limit: Optional[int] = None,
        generator: Optional[Generator] = None,
        render_mode: str = "human",
    ) -> None:
        """Instantiate an Cleaner environment.

        Args:
            grid_width: width of the grid.
            grid_height: height of the grid.
            num_agents: number of agents.
            step_limit: max number of steps in an episode. Defaults to grid_width * grid_height.
            render_mode: the mode for visualising the environment, can be "human" or "rgb_array".
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid_shape = (self.grid_width, self.grid_height)
        self.num_agents = num_agents
        self.step_limit = step_limit or (self.grid_width * self.grid_height)
        self.generator = generator or RandomGenerator(grid_width, grid_height)

        # Create viewer used for rendering
        self._env_viewer = CleanerViewer("Cleaner", render_mode)

    def __repr__(self) -> str:
        return (
            f"Cleaner(\n"
            f"\tgrid_width={self.grid_width!r},\n"
            f"\tgrid_height={self.grid_height!r},\n"
            f"\tnum_agents={self.num_agents!r}, \n"
            ")"
        )

    def observation_spec(self) -> specs.Spec:
        """Specification of the observation of the Cleaner environment.

        Returns:
            ObservationSpec containing the specifications for all observation fields:
                - grid: BoundedArray of int between 0 and 2 (inclusive),
                    same shape as the grid.
                - agent_locations_spec: BoundedArray of int, shape is (num_agents, 2).
                    Maximum value for the first column is grid_width,
                    and maximum value for the second is grid_height.
                - action_mask: BoundedArray of bool, shape is (num_agent, 4).
        """
        grid = specs.BoundedArray(self.grid_shape, int, 0, 2, "grid")
        agents_locations = specs.BoundedArray(
            (self.num_agents, 2), int, [0, 0], self.grid_shape, "agents_locations"
        )
        action_mask = specs.BoundedArray(
            (self.num_agents, 4), bool, False, True, "action_mask"
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            grid=grid,
            agents_locations=agents_locations,
            action_mask=action_mask,
        )

    def action_spec(self) -> specs.BoundedArray:
        """Specification of the actions for the Cleaner environment.

        Returns:
            BoundedArray (int) between 0 and 3 (inclusive) of shape (num_agents,).
        """
        return specs.BoundedArray((self.num_agents,), int, 0, 3, "actions")

    def render(self, state: State) -> None:
        """Render the given state of the environment.

        Args:
            state: `State` object containing the current environment state.
        """
        self._env_viewer.render(state)

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Reset the environment to its initial state.

        All the tiles except upper left are dirty, and the agents start in the upper left
        corner of the grid.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: State object corresponding to the new state of the environment after a reset.
            timestep: TimeStep object corresponding to the first timestep returned by the
                environment after a reset.
        """
        key, subkey = jax.random.split(key)

        # Agents start in upper left corner
        agents_locations = jnp.zeros((self.num_agents, 2), int)

        grid = self.generator(subkey)
        # Clean the tile in upper left corner
        grid = self._clean_tiles_containing_agents(grid, agents_locations)

        state = State(
            grid=grid,
            agents_locations=agents_locations,
            action_mask=self._compute_action_mask(grid, agents_locations),
            step_count=jnp.int32(0),
            key=key,
        )

        observation = self._observation_from_state(state)

        timestep = restart(observation)

        return state, timestep

    def step(
        self, state: State, actions: Action
    ) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of the environment's dynamics.

        If an action is invalid, the corresponding agent does not move and
        the episode terminates.

        Args:
            state: current environment state.
            actions: Jax array of shape (num_agents,). Each agent moves one step in
                the specified direction (0: up, 1: righ, 2: down, 3: left).

        Returns:
            state: State object corresponding to the next state of the environment.
            timestep: TimeStep object corresponding to the timestep returned by the environment.
        """
        are_actions_valid = self._are_actions_valid(actions, state.action_mask)

        agents_locations = self._update_agents_locations(
            state.agents_locations, actions, are_actions_valid
        )

        grid = self._clean_tiles_containing_agents(state.grid, agents_locations)

        prev_state = state

        state = State(
            agents_locations=agents_locations,
            grid=grid,
            action_mask=self._compute_action_mask(grid, agents_locations),
            step_count=state.step_count + 1,
            key=state.key,
        )

        reward = self._compute_reward(prev_state, state)

        observation = self._observation_from_state(state)

        done = self._should_terminate(state, are_actions_valid)

        # Return either a MID or a LAST timestep depending on done.
        timestep = jax.lax.cond(
            done,
            termination,
            transition,
            reward,
            observation,
        )

        return state, timestep

    def _compute_reward(self, prev_state: State, state: State) -> chex.Array:
        """Compute the reward by counting the number of tiles which changed from previous state.

        Since walls and dirty tiles do not change, counting the tiles which changed since previeous
        step is the same as counting the tiles which were cleaned.
        """
        return jnp.sum(prev_state.grid != state.grid, dtype=float)

    def _compute_action_mask(
        self, grid: chex.Array, agents_locations: chex.Array
    ) -> chex.Array:
        """Compute the action mask.

        An action is masked if it leads to a WALL or outside of the maze.
        """

        def is_move_valid(agent_location: chex.Array, move: chex.Array) -> chex.Array:
            y, x = agent_location + move
            return (
                (x >= 0)
                & (x < self.grid_width)
                & (y >= 0)
                & (y < self.grid_height)
                & (grid[y, x] != WALL)
            )

        # vmap over the moves and agents
        action_mask = jax.vmap(
            jax.vmap(is_move_valid, in_axes=(None, 0)), in_axes=(0, None)
        )(agents_locations, MOVES)

        return action_mask

    def _observation_from_state(self, state: State) -> Observation:
        """Create an observation from the state of the environment."""
        return Observation(
            grid=state.grid,
            agents_locations=state.agents_locations,
            action_mask=state.action_mask,
        )

    def _are_actions_valid(
        self, actions: Action, action_mask: chex.Array
    ) -> chex.Array:
        """Compute, for the action of each agent, whether said action is valid.

        Args:
            actions: Jax array containing the actions to validate.
            action_mask: Jax array containing the action mask.

        Returns:
            An array of booleans representing which agents took a valid action.
        """
        return action_mask[jnp.arange(self.num_agents), actions]

    def _update_agents_locations(
        self, prev_locations: chex.Array, actions: Action, actions_are_valid: chex.Array
    ) -> chex.Array:
        """Update the agents locations based on the actions if they are valid."""
        moves = jnp.where(actions_are_valid[:, None], MOVES[actions], 0)
        return prev_locations + moves

    def _clean_tiles_containing_agents(
        self, grid: chex.Array, agents_locations: chex.Array
    ) -> chex.Array:
        """Clean all tiles containing an agent."""
        return grid.at[agents_locations[:, 0], agents_locations[:, 1]].set(CLEAN)

    def _should_terminate(self, state: State, valid_actions: chex.Array) -> chex.Array:
        """Whether the episode should terminate from a given state.

        Returns True if:
            - An action was invalid.
            - There are no more dirty tiles left, i.e. all tiles have been cleaned.
            - The maximum number of steps (`self.step_limit`) is reached.
        Returns False otherwise.
        """
        return (
            ~valid_actions.all()
            | ~(state.grid == DIRTY).any()
            | (state.step_count >= self.step_limit)
        )
