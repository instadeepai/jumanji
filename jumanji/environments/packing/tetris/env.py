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
        - tetrominoe: jax array (int32) of shape (4, 4)
            representing the current tetrominoe sampled from the tetrominoe list.
        - action_mask: jax array (bool) of shape (4,  num_cols).
            For each tetrominoe there are 4 rotations, each one corresponds
            to a line in the action_mask.
            Mask of the joint action space: True if the action
            (x_position and rotation degree) is feasible
            for the current tetrominoe and grid state.
    - action: `Tuple`
        - x_position: int between 0 and num_cols - 1 (included).
        - rotation_degree: the degree to rotate the tetromino (0, 90, 180, or 270).
    - reward:
        The reward is given based on the number of lines cleared by the player.
    - episode termination:
        if the tetrominoe cannot be placed anymore (i.e., it hits the top of the grid).

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
        num_rows: int = 20,
        num_cols: int = 10,
        viewer: Optional[Viewer[State]] = None,
    ) -> None:
        """Instantiates a `Tetris` environment.

        Args:
            num_rows: number of rows of the 2D grid. Defaults to 20.
            num_cols: number of columns of the 2D grid. Defaults to 10.
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.padded_num_rows = num_rows + 3
        self.padded_num_cols = num_cols + 3
        self.tetrominoes_list = jnp.array(TETROMINOES_LIST)
        self.reward_list = jnp.array(REWARD_LIST, float)

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
            ]
        )

    def _calculate_action_mask(
        self, grid_padded: chex.Array, tetrominoe_index: int
    ) -> chex.Array:
        """Calculate the Mask for Legal Actions in the Game.

        Args:
            grid_padded: The container where the game takes place.
            tetrominoe: The block that will be placed within the grid_padded.

        Return:
            action_mask: jnp boolean array of size=(4 x 'self.num_cols').
            Each row of the matrix corresponds to a different possible
            rotation of the tetrominoe block,
            with the first row corresponding to a rotation of 0 degrees,
            the second row corresponding to a rotation of 90 degrees,
            the third row corresponding to a rotation of 180 degrees,
            and the fourth row corresponding to a rotation of 270 degrees.
        """
        all_rotations = self.tetrominoes_list[tetrominoe_index]
        all_rotations = jnp.squeeze(all_rotations)
        action_mask = [
            utils.tetrominoe_action_mask(grid_padded, all_rotations[i])
            for i in range(4)
        ]
        action_mask = jnp.array(action_mask)
        return jnp.squeeze(action_mask)

    def _rotate(self, rotation_degree: int, tetrominoe_index: int) -> chex.Array:
        """Calculate the Rotated Tetrominoe Matrix.
        This function calculates a matrix representation of a rotated "tetrominoe" block,
        given the desired rotation angle and an index to retrieve the block
        from a list of tetrominoes.

        Args:
            rotation_degree: The desired rotation angle, which can be 0, 90, 180, or 270 degrees.
            tetrominoe_index: An index used to retrieve a specific tetrominoe
            from the 'self.tetrominoes_list'.

        Return:
            rotated_tetrominoe (matrix): chex.array representation of the rotated tetrominoe block.
        """
        rotation_index = rotation_degree
        rotated_tetrominoe = self.tetrominoes_list[tetrominoe_index, rotation_index]
        rotated_tetrominoe = jnp.squeeze(rotated_tetrominoe)
        return rotated_tetrominoe

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment.

        Args:
            key: needed for generating new tetrominoes.

        Returns:
            state: State corresponding to the new state of the environment,
            timestep: TimeStep corresponding to the first timestep returned by the
            environment.
        """
        grid_padded = jnp.zeros(
            shape=(self.padded_num_rows, self.padded_num_cols), dtype=jnp.int32
        )
        tetrominoe, tetrominoe_index = utils.sample_tetrominoe_list(
            key, self.tetrominoes_list
        )

        action_mask = self._calculate_action_mask(grid_padded, tetrominoe_index)
        state = State(
            grid_padded=grid_padded,
            grid_padded_old=grid_padded,
            tetrominoe_index=tetrominoe_index,
            old_tetrominoe_rotated=tetrominoe,
            new_tetrominoe=tetrominoe,
            x_position=0,
            y_position=0,
            action_mask=action_mask,
            full_lines=jnp.full((self.num_rows), False),
            score=0,
            reward=0,
            key=key,
            is_reset=True,
        )

        observation = Observation(
            grid=grid_padded[: self.num_rows, : self.num_cols],
            tetrominoe=tetrominoe,
            action_mask=action_mask,
        )
        timestep = restart(observation=observation)
        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: `State` object containing the dynamics of the environment.
            action: `Array` containing the x_position and rotation_index of the tetrominoe.

        Returns:
            next_state: State corresponding to the next state of the environment,
            next_timestep: TimeStep corresponding to the timestep returned by the environment.
        """
        x_position, rotation_degree = action
        grid_padded = state.grid_padded
        action_mask = state.action_mask
        tetrominoe_index = state.tetrominoe_index
        # Generate new PRNG key
        key, subkey = jax.random.split(state.key)
        # Rotate tetrominoe.
        tetrominoe = self._rotate(rotation_degree, tetrominoe_index)
        # Place the tetrominoe in the selected place
        grid_padded, y_position = utils.place_tetrominoe(
            grid_padded, tetrominoe, x_position
        )
        # a line is full when it doesn't contain any 0.
        full_lines = jnp.all(grid_padded[:, : self.num_cols], axis=1)
        nbr_full_lines = sum(full_lines)
        grid_padded = utils.clean_lines(grid_padded, full_lines)
        # Generate new tetrominoe
        new_tetrominoe, tetrominoe_index = utils.sample_tetrominoe_list(
            key, self.tetrominoes_list
        )
        grid_padded_cliped = jnp.clip(grid_padded, a_max=1)
        action_mask = self._calculate_action_mask(grid_padded_cliped, tetrominoe_index)
        # The maximum should be bigger than 0.
        # In case the grid is empty the color should be set 0.
        color = jnp.array([1, grid_padded.max()])
        colored_tetrominoe = tetrominoe * jnp.max(color)
        reward = self.reward_list[nbr_full_lines]
        next_state = State(
            grid_padded=grid_padded,
            grid_padded_old=state.grid_padded,
            tetrominoe_index=tetrominoe_index,
            old_tetrominoe_rotated=colored_tetrominoe,
            new_tetrominoe=new_tetrominoe,
            x_position=x_position,
            y_position=y_position,
            action_mask=action_mask,
            full_lines=full_lines,
            score=state.score + reward,
            reward=reward,
            key=key,
            is_reset=False,
        )
        next_observation = Observation(
            grid=grid_padded_cliped[: self.num_rows, : self.num_cols],
            tetrominoe=new_tetrominoe,
            action_mask=action_mask,
        )
        next_timestep = jax.lax.cond(
            action_mask.sum() == 0,
            termination,
            transition,
            self.reward_list[nbr_full_lines],
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
             - tetrominoe: BoundedArray (bool) of shape (4, 4).
             - action_mask: BoundedArray (bool) of shape (NUM_ROTATIONS, num_cols).
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
            tetrominoe=specs.BoundedArray(
                shape=(4, 4),
                dtype=bool,
                minimum=False,
                maximum=True,
                name="tetrominoe",
            ),
            action_mask=specs.BoundedArray(
                shape=(NUM_ROTATIONS, self.num_cols),
                dtype=bool,
                minimum=False,
                maximum=True,
                name="action_mask",
            ),
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec. An action consists of two pieces of information:
        the x-position of the leftmost part of the tetrominoe and its amount of
        rotation (number of 90-degree rotations).

        Returns:
            The action spec, which is a `specs.MultiDiscreteArray` object.
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.array([self.num_cols, NUM_ROTATIONS]),
            name="action",
            dtype=jnp.int32,
        )

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

        return self._viewer.animate(states, interval, save_path)
