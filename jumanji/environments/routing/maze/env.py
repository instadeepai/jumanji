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

from typing import Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib.animation
from numpy.typing import NDArray

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.maze.constants import MOVES
from jumanji.environments.routing.maze.generator import Generator, RandomGenerator
from jumanji.environments.routing.maze.types import Observation, Position, State
from jumanji.environments.routing.maze.viewer import MazeEnvViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class Maze(Environment[State]):
    """A JAX implementation of a 2D Maze. The goal is to navigate the maze to find the target
    position.

    - observation:
        - agent_position: current 2D Position of agent.
        - target_position: 2D Position of target cell.
        - walls: jax array (bool) of shape (num_rows, num_cols)
            whose values are `True` where walls are and `False` for empty cells.
        - action_mask: array (bool) of shape (4,)
            defining the available actions in the current position.
        - step_count: jax array (int32) of shape ()
            step number of the episode.

    - action: jax array (int32) of shape () specifying which action to take: [0,1,2,3] correspond to
        [Up, Right, Down, Left]. If an invalid action is taken, i.e. there is a wall blocking the
        action, then no action (no-op) is taken.

    - reward: jax array (float32) of shape (): 1 if the target is reached, 0 otherwise.

    - episode termination (if any):
        - agent reaches the target position.
        - the time_limit is reached.

    - state: State:
        - agent_position: current 2D Position of agent.
        - target_position: 2D Position of target cell.
        - walls: jax array (bool) of shape (num_rows, num_cols)
            whose values are `True` where walls are and `False` for empty cells.
        - action_mask: array (bool) of shape (4,)
            defining the available actions in the current position.
        - step_count: jax array (int32) of shape ()
            step number of the episode.
        - key: random key (uint) of shape (2,).

    ```python
    from jumanji.environments import Maze
    env = Maze()
    key = jax.random.key(0)
    state, timestep = jax.jit(env.reset)(key)
    env.render(state)
    action = env.action_spec().generate_value()
    state, timestep = jax.jit(env.step)(state, action)
    env.render(state)
    ```
    """

    FIGURE_NAME = "Maze"
    FIGURE_SIZE = (6.0, 6.0)

    def __init__(
        self,
        generator: Optional[Generator] = None,
        time_limit: Optional[int] = None,
        viewer: Optional[Viewer[State]] = None,
    ) -> None:
        """Instantiates a `Maze` environment.

        Args:
            generator: `Generator` whose `__call__` instantiates an environment instance.
                Implemented options are [`ToyGenerator`, `RandomGenerator`].
                Defaults to `RandomGenerator` with `num_rows=10` and `num_cols=10`.
            time_limit: the time_limit of an episode, i.e. the maximum number of environment steps
                before the episode terminates. By default, `time_limit = num_rows * num_cols`.
            viewer: `Viewer` used for rendering. Defaults to `MazeEnvViewer` with "human" render
                mode.
        """
        self.generator = generator or RandomGenerator(num_rows=10, num_cols=10)
        self.num_rows = self.generator.num_rows
        self.num_cols = self.generator.num_cols
        self.shape = (self.num_rows, self.num_cols)
        self.time_limit = time_limit or self.num_rows * self.num_cols

        # Create viewer used for rendering
        self._viewer = viewer or MazeEnvViewer("Maze", render_mode="human")

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Maze environment:",
                f" - num_rows: {self.num_rows}",
                f" - num_cols: {self.num_cols}",
                f" - time_limit: {self.time_limit}",
                f" - generator: {self.generator}",
            ]
        )

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specifications of the observation of the `Maze` environment.

        Returns:
            Spec for the `Observation` whose fields are:
            - agent_position: tree of BoundedArray (int32) of shape ().
            - target_position: tree of BoundedArray (int32) of shape ().
            - walls: BoundedArray (bool) of shape (num_rows, num_cols).
            - step_count: Array (int32) of shape ().
            - action_mask: BoundedArray (bool) of shape (4,).
        """
        agent_position = specs.Spec(
            Position,
            "PositionSpec",
            row=specs.BoundedArray(
                (), jnp.int32, 0, self.num_rows - 1, "row_coordinate"
            ),
            col=specs.BoundedArray(
                (), jnp.int32, 0, self.num_cols - 1, "col_coordinate"
            ),
        )
        walls = specs.BoundedArray(
            shape=(self.num_rows, self.num_cols),
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

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment by calling the instance generator for a new instance.

        Args:
            key: random key used to reset the environment since it is stochastic.

        Returns:
            state: `State` object corresponding to the new state of the environment after a reset.
            timestep: `TimeStep` object corresponding the first timestep returned by the environment
                after a reset.
        """

        state = self.generator(key)

        # Create the action mask and update the state
        state.action_mask = self._compute_action_mask(state.walls, state.agent_position)

        # Generate the observation from the environment state.
        observation = self._observation_from_state(state)

        # Return a restart timestep whose step type is FIRST.
        timestep = restart(observation)

        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
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
        time_limit_exceeded = state.step_count >= self.time_limit

        done = no_actions_available | target_reached | time_limit_exceeded

        # Compute the reward.
        reward = jnp.array(state.agent_position == state.target_position, float)

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
                & (x < self.num_cols)
                & (y >= 0)
                & (y < self.num_rows)
                & ~(walls[x, y])
            )

        # vmap over the moves.
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
        """Creates an animated gif of the `Maze` environment based on the sequence of states.

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
