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
import matplotlib
from numpy.typing import NDArray

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.packing.tetris import utils
from jumanji.environments.packing.tetris.constants import (
    NUM_ROTATIONS,
    REWARD_LIST,
    TETROMINOES_LIST,
)
from jumanji.environments.packing.tetris.types import Observation, State
from jumanji.environments.packing.tetris.viewer import TetrisViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class Tetris(Environment[State]):
    """RL Environment for the game of Tetris.
    The environment has a grid where the player can place tetrominoes.
    The environment has the following characteristics:

    - observation: `Observation`
        - grid: jax array (int32) of shape (num_rows, num_cols)
            representing the current state of the grid.
        - tetromino: jax array (int32) of shape (4, 4)
            representing the current tetromino sampled from the tetromino list.
        - action_mask: jax array (bool) of shape (4,  num_cols).
            For each tetromino there are 4 rotations, each one corresponds
            to a line in the action_mask.
            Mask of the joint action space: True if the action
            (x_position and rotation degree) is feasible
            for the current tetromino and grid state.
    - action: multi discrete array of shape (2,)
        - rotation_index: The degree index determines the rotation of the
            tetromino: 0 corresponds to 0 degrees, 1 corresponds to 90 degrees,
            2 corresponds to 180 degrees, and 3 corresponds to 270 degrees.
        - x_position: int between 0 and num_cols - 1 (included).

    - reward:
        The reward is 0 if no lines was cleared by the action and a convex function of the number
        of cleared lines otherwise.

    - episode termination:
        if the tetromino cannot be placed anymore (i.e., it hits the top of the grid).

    ```python
    from jumanji.environments import Tetris
    env = Tetris()
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
        num_rows: int = 10,
        num_cols: int = 10,
        time_limit: int = 400,
        viewer: Optional[Viewer[State]] = None,
    ) -> None:
        """Instantiates a `Tetris` environment.

        Args:
            num_rows: number of rows of the 2D grid. Defaults to 10.
            num_cols: number of columns of the 2D grid. Defaults to 10.
            time_limit: time_limit of an episode, i.e. number of environment steps before
                the episode ends. Defaults to 400.
            viewer: `Viewer` used for rendering. Defaults to `TetrisViewer`.
        """
        if num_rows < 4:
            raise ValueError(
                f"The `num_rows` must be >= 4, but got num_rows={num_rows}"
            )
        if num_cols < 4:
            raise ValueError(
                f"The `num_cols` must be >= 4, but got num_cols={num_cols}"
            )
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.padded_num_rows = num_rows + 3
        self.padded_num_cols = num_cols + 3
        self.TETROMINOES_LIST = jnp.array(TETROMINOES_LIST, jnp.int32)
        self.reward_list = jnp.array(REWARD_LIST, float)
        self.time_limit = time_limit
        self._viewer = viewer or TetrisViewer(
            num_rows=self.num_rows,
            num_cols=self.num_cols,
        )

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Tetris environment:",
                f" - number rows: {self.num_rows}",
                f" - number columns: {self.num_cols}",
                f" - time_limit: {self.time_limit}",
            ]
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment.

        Args:
            key: needed for generating new tetrominoes.

        Returns:
            state: `State` corresponding to the new state of the environment,
            timestep: `TimeStep` corresponding to the first timestep returned by the
                environment.
        """
        grid_padded = jnp.zeros(
            shape=(self.padded_num_rows, self.padded_num_cols), dtype=jnp.int32
        )
        tetromino, tetromino_index = utils.sample_tetromino_list(
            key, self.TETROMINOES_LIST
        )

        action_mask = self._calculate_action_mask(grid_padded, tetromino_index)
        state = State(
            grid_padded=grid_padded,
            grid_padded_old=grid_padded,
            tetromino_index=tetromino_index,
            old_tetromino_rotated=tetromino,
            new_tetromino=tetromino,
            x_position=jnp.array(0, jnp.int32),
            y_position=jnp.array(0, jnp.int32),
            action_mask=action_mask,
            full_lines=jnp.full((self.num_rows + 3), False),
            score=jnp.array(0, float),
            reward=jnp.array(0, float),
            key=key,
            is_reset=True,
            step_count=jnp.array(0, jnp.int32),
        )

        observation = Observation(
            grid=grid_padded[: self.num_rows, : self.num_cols],
            tetromino=tetromino,
            action_mask=action_mask,
            step_count=jnp.array(0, jnp.int32),
        )
        timestep = restart(observation=observation)
        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: `State` object containing the dynamics of the environment.
            action: `chex.Array` containing the rotation_index and x_position of the tetromino.

        Returns:
            next_state: `State` corresponding to the next state of the environment,
            next_timestep: `TimeStep` corresponding to the timestep returned by the environment.
        """
        rotation_index, x_position = action
        tetromino_index = state.tetromino_index
        key, sample_key = jax.random.split(state.key)
        tetromino = self._rotate(rotation_index, tetromino_index)
        # Place the tetromino in the selected place
        grid_padded, y_position = utils.place_tetromino(
            state.grid_padded, tetromino, x_position
        )
        # A line is full when it doesn't contain any 0.
        full_lines = jnp.all(grid_padded[:, : self.num_cols] != 0, axis=1)
        nbr_full_lines = sum(full_lines)
        grid_padded = utils.clean_lines(grid_padded, full_lines)
        # Generate new tetromino
        new_tetromino, tetromino_index = utils.sample_tetromino_list(
            sample_key, self.TETROMINOES_LIST
        )
        grid_padded_cliped = jnp.clip(grid_padded, a_max=1)
        action_mask = self._calculate_action_mask(grid_padded_cliped, tetromino_index)
        # The maximum should be bigger than 0.
        # In case the grid is empty the color should be set 0.
        color = jnp.array([1, grid_padded.max()])
        colored_tetromino = tetromino * jnp.max(color)
        is_valid = state.action_mask[tuple(action)]
        reward = self.reward_list[nbr_full_lines] * is_valid
        step_count = state.step_count + 1
        next_state = State(
            grid_padded=grid_padded,
            grid_padded_old=state.grid_padded,
            tetromino_index=tetromino_index,
            old_tetromino_rotated=colored_tetromino,
            new_tetromino=new_tetromino,
            x_position=x_position,
            y_position=y_position,
            action_mask=action_mask,
            full_lines=full_lines,
            score=state.score + reward,
            reward=reward,
            key=key,
            is_reset=False,
            step_count=step_count,
        )
        next_observation = Observation(
            grid=grid_padded_cliped[: self.num_rows, : self.num_cols],
            tetromino=new_tetromino,
            action_mask=action_mask,
            step_count=jnp.array(0, jnp.int32),
        )

        tetris_completed = ~jnp.any(action_mask)
        done = tetris_completed | ~is_valid | (step_count >= self.time_limit)

        next_timestep = jax.lax.cond(
            done,
            termination,
            transition,
            reward,
            next_observation,
        )
        return next_state, next_timestep

    def render(self, state: State) -> Optional[NDArray]:
        """Render the given state of the environment.
        Args:
            state: `State` object containing the current environment state.
        """
        return self._viewer.render(state)

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specifications of the observation of the `Tetris` environment.

        Returns:
            Spec containing all the specifications for all the `Observation` fields:
             - grid: BoundedArray (jnp.int32) of shape (num_rows, num_cols).
             - tetromino: BoundedArray (bool) of shape (4, 4).
             - action_mask: BoundedArray (bool) of shape (NUM_ROTATIONS, num_cols).
             - step_count: DiscreteArray (num_values = time_limit) of shape ().
        """
        return specs.Spec(
            Observation,
            "ObservationSpec",
            grid=specs.BoundedArray(
                shape=(self.num_rows, self.num_cols),
                dtype=jnp.int32,
                minimum=0,
                maximum=1,
                name="grid",
            ),
            tetromino=specs.BoundedArray(
                shape=(4, 4),
                dtype=jnp.int32,
                minimum=0,
                maximum=1,
                name="tetromino",
            ),
            action_mask=specs.BoundedArray(
                shape=(NUM_ROTATIONS, self.num_cols),
                dtype=bool,
                minimum=False,
                maximum=True,
                name="action_mask",
            ),
            step_count=specs.DiscreteArray(
                self.time_limit, dtype=jnp.int32, name="step_count"
            ),
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec. An action consists of two pieces of information:
        the amount of rotation (number of 90-degree rotations) and the x-position of
        the leftmost part of the tetromino.

        Returns:
            The action spec, which is a `specs.MultiDiscreteArray` object.
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.array([NUM_ROTATIONS, self.num_cols]),
            name="action",
            dtype=jnp.int32,
        )

    def animate(
        self,
        states: Sequence[State],
        interval: int = 100,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of states.
        Args:
            states: sequence of `State` corresponding to subsequent timesteps.
            interval: delay between frames in milliseconds, default to 100.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.
        Returns:
            animation that can export to gif, mp4, or render with HTML.
        """

        return self._viewer.animate(states, interval, save_path)

    def _calculate_action_mask(
        self, grid_padded: chex.Array, tetromino_index: int
    ) -> chex.Array:
        """Calculate the mask for legal actions in the game.

        Args:
            grid_padded: the container where the game takes place.
            tetromino: the block that will be placed within the grid_padded.

        Return:
            action_mask: jnp boolean array of size=(4 x 'self.num_cols').
            Each row of the matrix corresponds to a different possible
            rotation of the tetromino block,
            with the first row corresponding to a rotation of 0 degrees,
            the second row corresponding to a rotation of 90 degrees,
            the third row corresponding to a rotation of 180 degrees,
            and the fourth row corresponding to a rotation of 270 degrees.
        """
        all_rotations = self.TETROMINOES_LIST[tetromino_index]
        action_mask = jax.vmap(utils.tetromino_action_mask, in_axes=(None, 0))(
            grid_padded, all_rotations
        )
        return action_mask

    def _rotate(self, rotation_index: int, tetromino_index: int) -> chex.Array:
        """Calculate the rotated tetromino matrix.
        This function calculates a matrix representation of a rotated "tetromino" block,
        given the desired rotation index and the tetromino index to retrieve the block
        from a list of tetrominoes.

        Args:
            rotation_index: the desired rotation index, which maps to 0, 90, 180, or 270 degrees.
            tetromino_index: an index used to retrieve a specific tetromino
                from the 'self.TETROMINOES_LIST'.

        Return:
            rotated_tetromino: array representation of the rotated tetromino block.
        """
        rotated_tetromino = self.TETROMINOES_LIST[tetromino_index, rotation_index]
        rotated_tetromino = jnp.squeeze(rotated_tetromino)
        return rotated_tetromino
