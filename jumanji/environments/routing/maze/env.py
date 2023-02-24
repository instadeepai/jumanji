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
from jumanji.environments.routing.maze.constants import MOVES
from jumanji.environments.routing.maze.types import Observation, Position, State
from jumanji.types import Action, TimeStep, restart, termination, transition


class Maze(Environment[State]):
    """A JAX implementation of a 2D Maze. The goal is to navigate the maze to find the target
       position.

    - observation:
        - agent_position: current 2D Position of agent.
        - target_position: 2D Position of target cell.
        - walls: array (bool) whose values are `True` where walls are and `False` for empty cells.
        - action_mask: array specifying which directions the agent can move in from its
            current position.
        - step_count: (int32) step number of the episode.

    - action: (int32) specifying which action to take: [0,1,2,3] correspond to
        [Up, Right, Down, Left]. If an invalid action is taken, i.e. there is a wall blocking the
        action, then no action (no-op) is taken.

    - reward: (float32) 1 if target reached, 0 otherwise.

    - episode termination (if any):
        - agent reaches the target position.
        - the horizon is reached.

    - state: State:
        - agent_position: current 2D Position of agent.
        - target_position: 2D Position of target cell.
        - walls: array (bool) whose values are `True` where walls are and `False` for empty cells.
        - action_mask: array (bool) of shape (4,)
            defining the available actions in the current position.
        - step_count: (int32), step number of the episode.
        - key: random key (uint) of shape (2,).
    """

    FIGURE_NAME = "Maze"
    FIGURE_SIZE = (6.0, 6.0)

    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        step_limit: Optional[int] = None,
    ):
        """Instantiates a Maze environment.

        Args:
            n_rows: number of rows, i.e. height of the maze.
            n_cols: number of columns, i.e. width of the maze.
            step_limit: the horizon of an episode, i.e. the maximum number of environment steps
                before the episode terminates. By default,
                `step_limit = 2 * self.n_rows * self.n_cols`.
        """

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.maze_shape = (self.n_rows, self.n_cols)
        self.step_limit = step_limit or 2 * self.n_rows * self.n_cols

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Maze environment:",
                f" - n_rows: {self.n_rows}",
                f" - n_cols: {self.n_cols}",
                f" - step_limit: {self.step_limit}",
            ]
        )

    def observation_spec(self) -> specs.Spec:
        """Specifications of the observation of the Maze environment.

        Returns: ObservationSpec containing all the specifications for all the Observation fields:
        observation_spec: ObservationSpec:
            - agent_position_spec : PositionSpec:
                - row_spec: specs.BoundedArray
                - col_spec: specs.BoundedArray
            - target_position_spec : PositionSpec:
                - row_spec: specs.BoundedArray
                - col_spec: specs.BoundedArray
            - walls_spec : specs.BoundedArray (bool) of shape (n_rows, n_cols),
            - step_count_spec : specs.Array int of shape ()
            - action_mask_spec : specs.BoundedArray (bool) of shape (4,).
        """
        agent_position = specs.Spec(
            Position,
            "PositionSpec",
            row=specs.BoundedArray((), jnp.int32, 0, self.n_rows - 1, "row_coordinate"),
            col=specs.BoundedArray((), jnp.int32, 0, self.n_cols - 1, "col_coordinate"),
        )
        walls = specs.BoundedArray(
            shape=(self.n_rows, self.n_cols),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="walls",
        )
        step_count = specs.Array((), jnp.int32, "step_count")
        action_mask = specs.BoundedArray(
            shape=(4,), dtype=bool, minimum=False, maximum=True, name="action_mask"
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            agent_position=agent_position,
            target_position=agent_position,
            walls=walls,
            step_count=step_count,
            action_mask=action_mask,
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec. 4 actions: [0,1,2,3] -> [Up, Right, Down, Left].

        Returns:
            action_spec: discrete action space with 4 values.
        """
        return specs.DiscreteArray(4, name="action")

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Resets the environment by calling the instance generator for a new instance.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: State object corresponding to the new state of the environment after a reset.
            timestep: TimeStep object corresponding the first timestep returned by the environment
                after a reset.
        """

        key, state_key, agent_key = jax.random.split(key, 3)

        # Generate a Toy instance.
        walls = jnp.ones((self.n_rows, self.n_cols), bool)
        walls = walls.at[0, :].set((False, True, False, False, False))
        walls = walls.at[1, :].set((False, True, False, True, True))
        walls = walls.at[2, :].set((False, True, False, False, False))
        walls = walls.at[3, :].set((False, False, False, True, True))
        walls = walls.at[4, :].set((False, False, False, False, False))

        agent_position = Position(row=0, col=0)
        target_position = Position(row=0, col=4)

        # Build the state.
        state = State(
            agent_position=agent_position,
            target_position=target_position,
            walls=walls,
            action_mask=self._compute_action_mask(walls, agent_position),
            key=key,
            step_count=jnp.int32(0),
        )

        # Generate the observation from the environment state.
        observation = self._observation_from_state(state)

        # Return a restart timestep whose step type is FIRST
        timestep = restart(observation)

        return state, timestep

    def step(self, state: State, action: Action) -> Tuple[State, TimeStep]:
        """
        Run one timestep of the environment's dynamics.

        If an action is invalid, the agent does not move, i.e. the episode does not
        automatically terminate.

        Args:
            state: State object containing the dynamics of the environment.
            action: (int32) specifying which action to take: [0,1,2,3] correspond to
                [Up, Right, Down, Left]. If an invalid action is taken, i.e. there is a wall
                blocking the action, then no action (no-op) is taken.

        Returns:
            state: the next state of the environment.
            timestep: the next timestep to be observed.
        """
        # If the chosen action is invalid, i.e. blocked by a wall, overwrite it to no-op.
        action = jax.lax.select(state.action_mask[action], action, 4)

        # Take the action in the environment:  up, right, down, or left
        # Remember the walls coordinates: (0,0) is top left.
        agent_position = jax.lax.switch(
            action,
            [
                lambda position: Position(position.row - 1, position.col),  # Up
                lambda position: Position(position.row, position.col + 1),  # Right
                lambda position: Position(position.row + 1, position.col),  # Down
                lambda position: Position(position.row, position.col - 1),  # Left
                lambda position: position,  # No-op
            ],
            state.agent_position,
        )

        # Generate action mask to keep in the state for the next step and
        # to provide to the agent in the observation.
        action_mask = self._compute_action_mask(state.walls, agent_position)

        # Build the state.
        state = State(
            agent_position=agent_position,
            target_position=state.target_position,
            walls=state.walls,
            action_mask=action_mask,
            key=state.key,
            step_count=state.step_count + 1,
        )
        # Generate the observation from the environment state.
        observation = self._observation_from_state(state)

        # Check if the episode terminates (i.e. done is True).
        no_actions_available = ~jnp.any(action_mask)
        target_reached = state.agent_position == state.target_position
        step_limit_exceeded = state.step_count >= self.step_limit

        done = no_actions_available | target_reached | step_limit_exceeded

        # Compute the reward.
        reward = jnp.float32(state.agent_position == state.target_position)

        # Return either a MID or a LAST timestep depending on done.
        timestep = jax.lax.cond(
            done,
            termination,
            transition,
            reward,
            observation,
        )
        return state, timestep

    def _compute_action_mask(
        self, walls: chex.Array, agent_position: Position
    ) -> chex.Array:
        """Compute the action mask.
        An action is considered invalid if it leads to a WALL or goes outside of the maze.
        """

        def is_move_valid(agent_position: Position, move: chex.Array) -> chex.Array:
            x, y = jnp.array([agent_position.row, agent_position.col]) + move
            return (
                (x >= 0)
                & (x < self.n_cols)
                & (y >= 0)
                & (y < self.n_rows)
                & ~(walls[x, y])
            )

        # vmap over the moves
        action_mask = jax.vmap(is_move_valid, in_axes=(None, 0))(agent_position, MOVES)

        return action_mask

    def _observation_from_state(self, state: State) -> Observation:
        """Create an observation from the state of the environment."""
        return Observation(
            agent_position=state.agent_position,
            target_position=state.target_position,
            walls=state.walls,
            step_count=state.step_count,
            action_mask=state.action_mask,
        )
