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

import os
from typing import Any, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib

from jumanji import Environment, specs
from jumanji.environments.logic.sudoku.constants import BOARD_WIDTH
from jumanji.environments.logic.sudoku.data import DATABASES
from jumanji.environments.logic.sudoku.generator import DatabaseGenerator, Generator
from jumanji.environments.logic.sudoku.reward import RewardFn, SparseRewardFn
from jumanji.environments.logic.sudoku.types import Observation, State
from jumanji.environments.logic.sudoku.utils import apply_action, get_action_mask
from jumanji.environments.logic.sudoku.viewer import SudokuViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class Sudoku(Environment[State]):
    """A JAX implementation of the sudoku game.

    - observation: `Observation`
        - board: jax array (int32) of shape (9,9):
            empty cells are represented by -1, and filled cells are represented by 0-8.
        - action_mask: jax array (bool) of shape (9,9,9):
            indicates which actions are valid.

    - action:
        multi discrete array containing the square to write a digit, and the digits
        to input.

    - reward: jax array (float32):
        1 at the end of the episode if the board is valid
        0 otherwise


    - state: `State`
        - board: jax array (int32) of shape (9,9):
            empty cells are represented by -1, and filled cells are represented by 0-8.

        - action_mask: jax array (bool) of shape (9,9,9):
            indicates which actions are valid (empty cells and valid digits).

        - key: jax array (int32) of shape (2,) used for seeding initial sudoku
            configuration.

    ```python
    from jumanji.environments import Sudoku
    env = Sudoku()
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
        viewer: Optional[Viewer[State]] = None,
    ):
        if generator is None:
            file_path = os.path.dirname(os.path.abspath(__file__))
            database_file = DATABASES["mixed"]
            database = jnp.load(os.path.join(file_path, "data", database_file))

        self._generator = generator or DatabaseGenerator(database=database)
        self._reward_fn = reward_fn or SparseRewardFn()
        self._viewer = viewer or SudokuViewer()

    def __repr__(self) -> str:
        return f"Sudoku(grid_size={BOARD_WIDTH}x{BOARD_WIDTH})"

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        state = self._generator(key)
        obs = Observation(board=state.board, action_mask=state.action_mask)
        timestep = restart(observation=obs)
        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        # check if action is valid
        invalid = ~state.action_mask[tuple(action)]
        updated_board = apply_action(action=action, board=state.board)
        updated_action_mask = get_action_mask(board=updated_board)

        # creating next state
        next_state = State(
            board=updated_board, action_mask=updated_action_mask, key=state.key
        )

        no_actions_available = ~jnp.any(updated_action_mask)

        # computing terminal condition
        done = invalid | no_actions_available
        reward = self._reward_fn(
            state=state, new_state=next_state, action=action, done=done
        )

        observation = Observation(board=updated_board, action_mask=updated_action_mask)

        timestep = jax.lax.cond(
            done,
            termination,
            transition,
            reward,
            observation,
        )

        return next_state, timestep

    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec containing the board and action_mask arrays.

        Returns:
            Spec containing all the specifications for all the `Observation` fields:
             - board: BoundedArray (jnp.int8) of shape (9,9).
             - action_mask: BoundedArray (bool) of shape (9,9,9).
        """

        board = specs.BoundedArray(
            shape=(BOARD_WIDTH, BOARD_WIDTH),
            minimum=-1,
            maximum=BOARD_WIDTH,
            dtype=jnp.int32,
            name="board",
        )

        action_mask = specs.BoundedArray(
            shape=(BOARD_WIDTH, BOARD_WIDTH, BOARD_WIDTH),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask",
        )

        return specs.Spec(
            Observation, "ObservationSpec", board=board, action_mask=action_mask
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec. An action is composed of 3 integers: the row index,
        the column index and the value to be placed in the cell.

        Returns:
            action_spec: `MultiDiscreteArray` object.
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.array([BOARD_WIDTH, BOARD_WIDTH, BOARD_WIDTH]),
            name="action",
            dtype=jnp.int32,
        )

    def render(self, state: State) -> Any:
        """Renders the current state of the sudoku.

        Args:
            state: the current state to be rendered.
        """
        return self._viewer.render(state=state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Creates an animated gif of the board based on the sequence of states.

        Args:
            states: a list of `State` objects representing the sequence of states.
            interval: the delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            animation.FuncAnimation: the animation object that was created.
        """
        return self._viewer.animate(
            states=states, interval=interval, save_path=save_path
        )
