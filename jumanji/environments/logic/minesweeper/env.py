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

from typing import List, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib.animation
import matplotlib.pyplot as plt

import jumanji.environments
from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.logic.minesweeper.constants import (
    COLOUR_MAPPING,
    PATCH_SIZE,
    UNEXPLORED_ID,
)
from jumanji.environments.logic.minesweeper.done import DefaultDoneFn, DoneFn
from jumanji.environments.logic.minesweeper.reward import DefaultRewardFn, RewardFn
from jumanji.environments.logic.minesweeper.types import Observation, State
from jumanji.environments.logic.minesweeper.utils import (
    count_adjacent_mines,
    create_flat_mine_locations,
    explored_mine,
)
from jumanji.types import TimeStep, restart, termination, transition


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
        multi discrete array containing the square to explore (height and width).

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
        num_rows: int = 10,
        num_cols: int = 10,
        num_mines: int = 10,
        reward_function: Optional[RewardFn] = None,
        done_function: Optional[DoneFn] = None,
        color_mapping: Optional[List[str]] = None,
    ):
        """Instantiate a `Minesweeper` environment.

        Args:
            num_rows: number of rows, i.e. height of the board. Defaults to 10.
            num_cols: number of columns, i.e. width of the board. Defaults to 10.
            num_mines: number of mines on the board. Defaults to 10.
            reward_function: `RewardFn` whose `__call__` method computes the reward of an
                environment transition based on the given current state and selected action.
                Implemented options are [`DefaultRewardFn`]. Defaults to `DefaultRewardFn`.
            done_function: `DoneFn` whose `__call__` method computes the done signal given the
                current state, action taken, and next state.
                Implemented options are [`DefaultDoneFn`]. Defaults to `DefaultDoneFn`.
            color_mapping: colour map used for rendering.
        """
        if num_rows <= 1 or num_cols <= 1:
            raise ValueError(
                f"Should make a board of height and width greater than 1, "
                f"got num_rows={num_rows}, num_cols={num_cols}"
            )
        if num_mines < 0 or num_mines >= num_rows * num_cols:
            raise ValueError(
                f"Number of mines should be constrained between 0 and the size of the board, "
                f"got {num_mines}"
            )
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_mines = num_mines
        self.reward_function = reward_function or DefaultRewardFn()
        self.done_function = done_function or DefaultDoneFn()

        self.cmap = color_mapping if color_mapping else COLOUR_MAPPING
        self.figure_name = f"{num_rows}x{num_cols} Minesweeper"
        self.figure_size = (6.0, 6.0)

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment.

        Args:
            key: needed for placing mines.

        Returns:
            state: `State` corresponding to the new state of the environment,
            timestep: `TimeStep` corresponding to the first timestep returned by the
                environment.
        """
        key, sample_key = jax.random.split(key)
        board = jnp.full(
            shape=(self.num_rows, self.num_cols),
            fill_value=UNEXPLORED_ID,
            dtype=jnp.int32,
        )
        step_count = jnp.array(0, jnp.int32)
        flat_mine_locations = create_flat_mine_locations(
            key=sample_key,
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            num_mines=self.num_mines,
        )
        state = State(
            board=board,
            step_count=step_count,
            key=key,
            flat_mine_locations=flat_mine_locations,
        )
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
        board = state.board
        action_height, action_width = action
        board = board.at[action_height, action_width].set(
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

    def render(self, state: State) -> None:
        """Render the given environment state using matplotlib.

        Args:
            state: environment state to be rendered.

        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        self._draw(ax, state)
        self._update_display(fig)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of environment states.

        Args:
            states: sequence of environment states corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation object that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig, ax = self._get_fig_ax()
        plt.tight_layout()
        plt.close(fig)

        def make_frame(state_index: int) -> None:
            state = states[state_index]
            self._draw(ax, state)

        # Create the animation object.
        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=len(states),
            interval=interval,
        )

        # Save the animation as a GIF.
        if save_path:
            self._animation.save(save_path)

        return self._animation

    def close(self) -> None:
        """Perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        plt.close(self.figure_name)

    def _get_fig_ax(self) -> Tuple[plt.Figure, plt.Axes]:
        exists = plt.fignum_exists(self.figure_name)
        if exists:
            fig = plt.figure(self.figure_name)
            ax = fig.get_axes()[0]
        else:
            fig = plt.figure(self.figure_name, figsize=self.figure_size)
            plt.suptitle(self.figure_name)
            plt.tight_layout()
            if not plt.isinteractive():
                fig.show()
            ax = fig.add_subplot()
        return fig, ax

    def _draw(self, ax: plt.Axes, state: State) -> None:
        ax.clear()
        ax.set_xticks(jnp.arange(-0.5, self.num_cols - 1, 1))
        ax.set_yticks(jnp.arange(-0.5, self.num_rows - 1, 1))
        ax.tick_params(
            top=False,
            bottom=False,
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False,
            labeltop=False,
            labelright=False,
        )
        background = jnp.ones_like(state.board)
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                background = self._render_grid_square(
                    state=state, ax=ax, i=i, j=j, background=background
                )
        ax.imshow(background, cmap="gray", vmin=0, vmax=1)
        ax.grid(color="black", linestyle="-", linewidth=2)

    def _render_grid_square(
        self, state: State, ax: plt.Axes, i: int, j: int, background: chex.Array
    ) -> chex.Array:
        board_value = state.board[i, j]
        if board_value != UNEXPLORED_ID:
            if explored_mine(state=state, action=jnp.array([i, j], dtype=jnp.int32)):
                background = background.at[i, j].set(0)
            else:
                ax.text(
                    j,
                    i,
                    str(board_value),
                    color=self.cmap[board_value],
                    ha="center",
                    va="center",
                    fontsize="xx-large",
                )
        return background

    def _update_display(self, fig: plt.Figure) -> None:
        if plt.isinteractive():
            # Required to update render when using Jupyter Notebook.
            fig.canvas.draw()
            if jumanji.environments.is_colab():
                plt.show(self.figure_name)
        else:
            # Required to update render when not using Jupyter Notebook.
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    def _clear_display(self) -> None:
        if jumanji.environments.is_colab():
            import IPython.display

            IPython.display.clear_output(True)
