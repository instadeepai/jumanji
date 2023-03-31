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
from jumanji.environments.logic.minesweeper.constants import PATCH_SIZE, UNEXPLORED_ID
from jumanji.environments.logic.minesweeper.done import DefaultDoneFn, DoneFn
from jumanji.environments.logic.minesweeper.generator import (
    Generator,
    UniformSamplingGenerator,
)
from jumanji.environments.logic.minesweeper.reward import DefaultRewardFn, RewardFn
from jumanji.environments.logic.minesweeper.types import Observation, State
from jumanji.environments.logic.minesweeper.utils import count_adjacent_mines
from jumanji.environments.logic.minesweeper.viewer import MinesweeperViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class Minesweeper(Environment[State]):
    """A JAX implementation of the minesweeper game.

    - observation: `Observation`
        - board: jax array (int32) of shape (num_rows, num_cols):
            each cell contains -1 if not yet explored, or otherwise the number of mines in
            the 8 adjacent squares.
        - action_mask: jax array (bool) of shape (num_rows, num_cols):
            indicates which actions are valid (not yet explored squares).
        - num_mines: jax array (int32) of shape `()`, indicates the number of mines to locate.
        - step_count: jax array (int32) of shape ():
            specifies how many timesteps have elapsed since environment reset.

    - action:
        multi discrete array containing the square to explore (row and col).

    - reward: jax array (float32):
        Configurable function of state and action. By default:
            1 for every timestep where a valid action is chosen that doesn't reveal a mine,
            0 for revealing a mine or selecting an already revealed square
                (and terminate the episode).

    - episode termination:
        Configurable function of state, next_state, and action. By default:
            Stop the episode if a mine is explored, an invalid action is selected
            (exploring an already explored square), or the board is solved.

    - state: `State`
        - board: jax array (int32) of shape (num_rows, num_cols):
            each cell contains -1 if not yet explored, or otherwise the number of mines in
            the 8 adjacent squares.
        - step_count: jax array (int32) of shape ():
            specifies how many timesteps have elapsed since environment reset.
        - flat_mine_locations: jax array (int32) of shape (num_rows * num_cols,):
            indicates the (flat) locations of all the mines on the board.
            Will be of length num_mines.
        - key: jax array (int32) of shape (2,) used for seeding the sampling of mine placement
            on reset.

    ```python
    from jumanji.environments import Minesweeper
    env = Minesweeper()
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
        reward_function: Optional[RewardFn] = None,
        done_function: Optional[DoneFn] = None,
        viewer: Optional[Viewer[State]] = None,
    ):
        """Instantiate a `Minesweeper` environment.

        Args:
            generator: `Generator` to generate problem instances on environment reset.
                Implemented options are [`SamplingGenerator`]. Defaults to `SamplingGenerator`.
                The generator will have attributes:
                    - num_rows: number of rows, i.e. height of the board. Defaults to 10.
                    - num_cols: number of columns, i.e. width of the board. Defaults to 10.
                    - num_mines: number of mines generated. Defaults to 10.
            reward_function: `RewardFn` whose `__call__` method computes the reward of an
                environment transition based on the given current state and selected action.
                Implemented options are [`DefaultRewardFn`]. Defaults to `DefaultRewardFn`, giving
                a reward of 1.0 for revealing an empty square, 0.0 for revealing a mine, and
                0.0 for an invalid action (selecting an already revealed square).
            done_function: `DoneFn` whose `__call__` method computes the done signal given the
                current state, action taken, and next state.
                Implemented options are [`DefaultDoneFn`]. Defaults to `DefaultDoneFn`, ending the
                episode on solving the board, revealing a mine, or picking an invalid action.
            viewer: `Viewer` to support rendering and animation methods.
                Implemented options are [`MinesweeperViewer`]. Defaults to `MinesweeperViewer`.
        """
        self.reward_function = reward_function or DefaultRewardFn(
            revealed_empty_square_reward=1.0,
            revealed_mine_reward=0.0,
            invalid_action_reward=0.0,
        )
        self.done_function = done_function or DefaultDoneFn()
        self.generator = generator or UniformSamplingGenerator(
            num_rows=10, num_cols=10, num_mines=10
        )
        self.num_rows = self.generator.num_rows
        self.num_cols = self.generator.num_cols
        self.num_mines = self.generator.num_mines
        self._viewer = viewer or MinesweeperViewer(
            num_rows=self.num_rows, num_cols=self.num_cols
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment.

        Args:
            key: needed for placing mines.

        Returns:
            state: `State` corresponding to the new state of the environment,
            timestep: `TimeStep` corresponding to the first timestep returned by the
                environment.
        """
        state = self.generator(key)
        observation = self._state_to_observation(state=state)
        timestep = restart(observation=observation)
        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: `State` object containing the dynamics of the environment.
            action: `Array` containing the row and column of the square to be explored.

        Returns:
            next_state: `State` corresponding to the next state of the environment,
            next_timestep: `TimeStep` corresponding to the timestep returned by the environment.
        """
        board = state.board.at[tuple(action)].set(
            count_adjacent_mines(state=state, action=action)
        )
        step_count = state.step_count + 1
        next_state = State(
            board=board,
            step_count=step_count,
            key=state.key,
            flat_mine_locations=state.flat_mine_locations,
        )
        reward = self.reward_function(state, action)
        done = self.done_function(state, next_state, action)
        next_observation = self._state_to_observation(state=next_state)
        next_timestep = jax.lax.cond(
            done,
            termination,
            transition,
            reward,
            next_observation,
        )
        return next_state, next_timestep

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specifications of the observation of the `Minesweeper` environment.

        Returns:
            Spec for the `Observation` whose fields are:
             - board: BoundedArray (int32) of shape (num_rows, num_cols).
             - action_mask: BoundedArray (bool) of shape (num_rows, num_cols).
             - num_mines: BoundedArray (int32) of shape ().
             - step_count: BoundedArray (int32) of shape ().
        """
        board = specs.BoundedArray(
            shape=(self.num_rows, self.num_cols),
            dtype=jnp.int32,
            minimum=-1,
            maximum=PATCH_SIZE * PATCH_SIZE - 1,
            name="board",
        )
        action_mask = specs.BoundedArray(
            shape=(self.num_rows, self.num_cols),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask",
        )
        num_mines = specs.BoundedArray(
            shape=(),
            dtype=jnp.int32,
            minimum=0,
            maximum=self.num_rows * self.num_cols - 1,
            name="num_mines",
        )
        step_count = specs.BoundedArray(
            shape=(),
            dtype=jnp.int32,
            minimum=0,
            maximum=self.num_rows * self.num_cols - self.num_mines,
            name="step_count",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            board=board,
            action_mask=action_mask,
            num_mines=num_mines,
            step_count=step_count,
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec.
        An action consists of the height and width of the square to be explored.

        Returns:
            action_spec: `specs.MultiDiscreteArray` object.
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.array([self.num_rows, self.num_cols], jnp.int32),
            name="action",
            dtype=jnp.int32,
        )

    def _state_to_observation(self, state: State) -> Observation:
        return Observation(
            board=state.board,
            action_mask=jnp.equal(state.board, UNEXPLORED_ID),
            num_mines=jnp.array(self.num_mines, jnp.int32),
            step_count=state.step_count,
        )

    def render(self, state: State) -> Optional[NDArray]:
        """Renders the current state of the board.

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

    def close(self) -> None:
        """Perform any necessary cleanup.
        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        self._viewer.close()
