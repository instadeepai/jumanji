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

from typing import Any, Dict, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib
from numpy.typing import NDArray

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.cleaner.constants import CLEAN, DIRTY, MOVES, WALL
from jumanji.environments.routing.cleaner.generator import Generator, RandomGenerator
from jumanji.environments.routing.cleaner.types import Observation, State
from jumanji.environments.routing.cleaner.viewer import CleanerViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class Cleaner(Environment[State]):
    """A JAX implementation of the 'Cleaner' game where multiple agents have to clean all tiles of
    a maze.

    - observation: `Observation`
        - grid: jax array (int32) of shape (num_rows, num_cols)
            contains the state of the board: 0 for dirty tile, 1 for clean tile, 2 for wall.
        - agents_locations: jax array (int32) of shape (num_agents, 2)
            contains the location of each agent on the board.
        - action_mask: jax array (bool) of shape (num_agents, 4)
            indicates for each agent if each of the four actions (up, right, down, left) is allowed.
        - step_count: (int32)
            the number of step since the beginning of the episode.

    - action: jax array (int32) of shape (num_agents,)
        the action for each agent: (0: up, 1: right, 2: down, 3: left)

    - reward: jax array (float) of shape ()
        +1 every time a tile is cleaned and a configurable penalty (-0.5 by default) for
        each timestep.

    - episode termination:
        - All tiles are clean.
        - The number of steps is greater than the limit.
        - An invalid action is selected for any of the agents.

    - state: `State`
        - grid: jax array (int32) of shape (num_rows, num_cols)
            contains the current state of the board: 0 for dirty tile, 1 for clean tile, 2 for wall.
        - agents_locations: jax array (int32) of shape (num_agents, 2)
            contains the location of each agent on the board.
        - action_mask: jax array (bool) of shape (num_agents, 4)
            indicates for each agent if each of the four actions (up, right, down, left) is allowed.
        - step_count: jax array (int32) of shape ()
            the number of steps since the beginning of the episode.
        - key: jax array (uint) of shape (2,)
            jax random generation key. Ignored since the environment is deterministic.

    ```python
    from jumanji.environments import Cleaner
    env = Cleaner()
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
        time_limit: Optional[int] = None,
        penalty_per_timestep: float = 0.5,
        viewer: Optional[Viewer[State]] = None,
    ) -> None:
        """Instantiates a `Cleaner` environment.

        Args:
            num_agents: number of agents. Defaults to 3.
            time_limit: max number of steps in an episode. Defaults to `num_rows * num_cols`.
            generator: `Generator` whose `__call__` instantiates an environment instance.
                Implemented options are [`RandomGenerator`]. Defaults to `RandomGenerator` with
                `num_rows=10`, `num_cols=10` and `num_agents=3`.
            viewer: `Viewer` used for rendering. Defaults to `CleanerViewer` with "human" render
                mode.
            penalty_per_timestep: the penalty returned at each timestep in the reward.
        """
        self.generator = generator or RandomGenerator(
            num_rows=10, num_cols=10, num_agents=3
        )
        self.num_agents = self.generator.num_agents
        self.num_rows = self.generator.num_rows
        self.num_cols = self.generator.num_cols
        self.grid_shape = (self.num_rows, self.num_cols)
        self.time_limit = time_limit or (self.num_rows * self.num_cols)
        self.penalty_per_timestep = penalty_per_timestep

        # Create viewer used for rendering
        self._viewer = viewer or CleanerViewer("Cleaner", render_mode="human")

    def __repr__(self) -> str:
        return (
            f"Cleaner(\n"
            f"\tnum_rows={self.num_rows!r},\n"
            f"\tnum_cols={self.num_cols!r},\n"
            f"\tnum_agents={self.num_agents!r}, \n"
            f"\tgenerator={self.generator!r}, \n"
            ")"
        )

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specification of the observation of the `Cleaner` environment.

        Returns:
            Spec for the `Observation`, consisting of the fields:
                - grid: BoundedArray (int32) of shape (num_rows, num_cols). Values
                    are between 0 and 2 (inclusive).
                - agent_locations_spec: BoundedArray (int32) of shape (num_agents, 2).
                    Maximum value for the first column is num_rows, and maximum value
                    for the second is num_cols.
                - action_mask: BoundedArray (bool) of shape (num_agent, 4).
                - step_count: BoundedArray (int32) of shape ().
        """
        grid = specs.BoundedArray(self.grid_shape, jnp.int32, 0, 2, "grid")
        agents_locations = specs.BoundedArray(
            (self.num_agents, 2), jnp.int32, [0, 0], self.grid_shape, "agents_locations"
        )
        action_mask = specs.BoundedArray(
            (self.num_agents, 4), bool, False, True, "action_mask"
        )
        step_count = specs.BoundedArray((), jnp.int32, 0, self.time_limit, "step_count")
        return specs.Spec(
            Observation,
            "ObservationSpec",
            grid=grid,
            agents_locations=agents_locations,
            action_mask=action_mask,
            step_count=step_count,
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Specification of the action for the `Cleaner` environment.

        Returns:
            action_spec: a `specs.MultiDiscreteArray` spec.
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.full(self.num_agents, 4, jnp.int32),
            dtype=jnp.int32,
            name="action_spec",
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Reset the environment to its initial state.

        All the tiles except upper left are dirty, and the agents start in the upper left
        corner of the grid.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: `State` object corresponding to the new state of the environment after a reset.
            timestep: `TimeStep` object corresponding to the first timestep returned by the
                environment after a reset.
        """
        # Agents start in upper left corner
        agents_locations = jnp.zeros((self.num_agents, 2), int)

        state = self.generator(key)

        # Create the action mask and update the state
        state.action_mask = self._compute_action_mask(state.grid, agents_locations)

        observation = self._observation_from_state(state)

        extras = self._compute_extras(state)
        timestep = restart(observation, extras)

        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of the environment's dynamics.

        If an action is invalid, the corresponding agent does not move and
        the episode terminates.

        Args:
            state: current environment state.
            action: Jax array of shape (num_agents,). Each agent moves one step in
                the specified direction (0: up, 1: right, 2: down, 3: left).

        Returns:
            state: `State` object corresponding to the next state of the environment.
            timestep: `TimeStep` object corresponding to the timestep returned by the environment.
        """
        is_action_valid = self._is_action_valid(action, state.action_mask)

        agents_locations = self._update_agents_locations(
            state.agents_locations, action, is_action_valid
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

        done = self._should_terminate(state, is_action_valid)

        extras = self._compute_extras(state)
        # Return either a MID or a LAST timestep depending on done.
        timestep = jax.lax.cond(
            done,
            lambda reward, observation, extras: termination(
                reward=reward,
                observation=observation,
                extras=extras,
            ),
            lambda reward, observation, extras: transition(
                reward=reward,
                observation=observation,
                extras=extras,
            ),
            reward,
            observation,
            extras,
        )

        return state, timestep

    def render(self, state: State) -> Optional[NDArray]:
        """Render the given state of the environment.

        Args:
            state: `State` object containing the current environment state.
        """
        return self._viewer.render(state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Creates an animated gif of the `Cleaner` environment based on the sequence of states.

        Args:
            states: sequence of environment states corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            animation.FuncAnimation: the animation object that was created.
        """
        return self._viewer.animate(states, interval, save_path)

    def close(self) -> None:
        """Perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        self._viewer.close()

    def _compute_reward(self, prev_state: State, state: State) -> chex.Array:
        """Compute the reward by counting the number of tiles which changed from previous state.

        Since walls and dirty tiles do not change, counting the tiles which changed since previeous
        step is the same as counting the tiles which were cleaned.
        """
        return (
            jnp.sum(prev_state.grid != state.grid, dtype=float)
            - self.penalty_per_timestep
        )

    def _compute_action_mask(
        self, grid: chex.Array, agents_locations: chex.Array
    ) -> chex.Array:
        """Compute the action mask.

        An action is masked if it leads to a WALL or out of the maze.
        """

        def is_move_valid(agent_location: chex.Array, move: chex.Array) -> chex.Array:
            y, x = agent_location + move
            return (
                (x >= 0)
                & (x < self.num_rows)
                & (y >= 0)
                & (y < self.num_cols)
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
            step_count=state.step_count,
        )

    def _is_action_valid(
        self, action: chex.Array, action_mask: chex.Array
    ) -> chex.Array:
        """Compute, for the action of each agent, whether said action is valid.

        Args:
            action: Jax array containing the action to validate for each agent.
            action_mask: Jax array containing the action mask.

        Returns:
            An array of booleans representing which agents took a valid action.
        """
        return action_mask[jnp.arange(self.num_agents), action]

    def _update_agents_locations(
        self,
        prev_locations: chex.Array,
        action: chex.Array,
        action_is_valid: chex.Array,
    ) -> chex.Array:
        """Update the agents locations based on the actions if they are valid."""
        moves = jnp.where(action_is_valid[:, None], MOVES[action], 0)
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
            - The maximum number of steps (`self.time_limit`) is reached.
        Returns False otherwise.
        """
        return (
            ~valid_actions.all()
            | ~(state.grid == DIRTY).any()
            | (state.step_count >= self.time_limit)
        )

    def _compute_extras(self, state: State) -> Dict[str, Any]:
        grid = state.grid
        ratio_dirty_tiles = jnp.sum(grid == DIRTY) / jnp.sum(grid != WALL)
        num_dirty_tiles = jnp.sum(grid == DIRTY)
        return {
            "ratio_dirty_tiles": ratio_dirty_tiles,
            "num_dirty_tiles": num_dirty_tiles,
        }
