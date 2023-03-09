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
import matplotlib.animation
import matplotlib.pyplot as plt
from chex import PRNGKey
from jax import lax
from jax import numpy as jnp

import jumanji.environments
from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.logic.minesweeper.constants import (
    COLOUR_MAPPING,
    DEFAULT_BOARD_HEIGHT,
    DEFAULT_BOARD_WIDTH,
    DEFAULT_NUM_MINES,
    PATCH_SIZE,
    UNEXPLORED_ID,
)
from jumanji.environments.logic.minesweeper.done_functions import (
    DefaultDoneFunction,
    DoneFunction,
)
from jumanji.environments.logic.minesweeper.reward_functions import (
    DefaultRewardFunction,
    RewardFunction,
)
from jumanji.environments.logic.minesweeper.types import Observation, State
from jumanji.environments.logic.minesweeper.utils import (
    count_adjacent_mines,
    create_flat_mine_locations,
    explored_mine,
)
from jumanji.types import Action, TimeStep, restart, termination, transition


def state_to_observation(state: State, num_mines: int) -> Observation:
    return Observation(
        board=state.board,
        action_mask=jnp.equal(state.board, UNEXPLORED_ID),
        num_mines=jnp.array(num_mines, jnp.int32),
        step_count=state.step_count,
    )


class Minesweeper(Environment[State]):
    """A JAX implementation of the minesweeper game.

    - observation: `Observation`
        - board: jax array (int32) of shape (board_height, board_width):
            each cell contains -1 if not yet explored, or otherwise the number of mines in
            the 8 adjacent squares.
        - action_mask: jax array (bool) of shape (board_height, board_width):
            indicates which actions are valid (not yet explored squares).
        - num_mines: the number of mines to find, which can be read from the env.
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
        - board: as in observation.
        - step_count: as in observation.
        - flat_mine_locations: jax array (int32) of shape (board_height * board_width):
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
        board_height: int = DEFAULT_BOARD_HEIGHT,
        board_width: int = DEFAULT_BOARD_WIDTH,
        reward_function: Optional[RewardFunction] = None,
        done_function: Optional[DoneFunction] = None,
        num_mines: int = DEFAULT_NUM_MINES,
        color_mapping: Optional[List[str]] = None,
    ):
        if board_height <= 1 or board_width <= 1:
            raise ValueError(
                f"Should make a board of height and width greater than 1, "
                f"got height={board_height}, width={board_width}"
            )
        if num_mines < 0 or num_mines >= board_height * board_width:
            raise ValueError(
                f"Number of mines should be constrained between 0 and the size of the board, "
                f"got {num_mines}"
            )
        self.board_height = board_height
        self.board_width = board_width
        self.num_mines = num_mines
        self.reward_function = reward_function or DefaultRewardFunction()
        self.done_function = done_function or DefaultDoneFunction()

        self.cmap = color_mapping if color_mapping else COLOUR_MAPPING
        self.figure_name = f"{board_height}x{board_width} Minesweeper"
        self.figure_size = (6.0, 6.0)

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment.

        Args:
            key: needed for placing mines.

        Returns:
            state: `State` corresponding to the new state of the environment,
            timestep: `TimeStep` corresponding to the first timestep returned by the
                environment.
        """
        board = jnp.full(
            shape=(self.board_height, self.board_width),
            fill_value=UNEXPLORED_ID,
            dtype=jnp.int32,
        )
        step_count = jnp.array(0, jnp.int32)
        flat_mine_locations = create_flat_mine_locations(
            key=key,
            board_height=self.board_height,
            board_width=self.board_width,
            num_mines=self.num_mines,
        )
        state = State(
            board=board,
            step_count=step_count,
            key=key,
            flat_mine_locations=flat_mine_locations,
        )
        observation = state_to_observation(state=state, num_mines=self.num_mines)
        timestep = restart(observation=observation)
        return state, timestep

    def step(self, state: State, action: Action) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: `State` object containing the dynamics of the environment.
            action: `Array` containing the height and width of the square to be explored.

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
        next_observation = state_to_observation(
            state=next_state, num_mines=self.num_mines
        )
        next_timestep = lax.cond(
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
            Spec containing all the specifications for all the `Observation` fields:
             - board: BoundedArray (jnp.int32) of shape (board_height, board_width).
             - action_mask: BoundedArray (bool) of shape (board_height, board_width).
             - num_mines: BoundedArray (jnp.int32) of shape ().
             - step_count: BoundedArray (jnp.int32) of shape ().
        """
        return specs.Spec(
            Observation,
            "ObservationSpec",
            board=specs.BoundedArray(
                shape=(self.board_height, self.board_width),
                dtype=jnp.int32,
                minimum=-1,
                maximum=PATCH_SIZE * PATCH_SIZE - 1,
                name="board",
            ),
            action_mask=specs.BoundedArray(
                shape=(self.board_height, self.board_width),
                dtype=jnp.bool_,
                minimum=False,
                maximum=True,
                name="action_mask",
            ),
            num_mines=specs.BoundedArray(
                shape=(),
                dtype=jnp.int32,
                minimum=0,
                maximum=self.board_height * self.board_width - 1,
                name="num_mines",
            ),
            step_count=specs.BoundedArray(
                shape=(),
                dtype=jnp.int32,
                minimum=0,
                maximum=self.board_height * self.board_width - self.num_mines,
                name="step_count",
            ),
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec.
        An action consists of the height and width of the square to be explored.

        Returns:
            action_spec: `specs.MultiDiscreteArray` object.
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.array([self.board_height, self.board_width]),
            name="action",
            dtype=jnp.int32,
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
        save: bool = False,
        path: str = "./minesweeper.gif",
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of environment states.

        Args:
            states: sequence of environment states corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save: whether to save the animation to a file.
            path: the path to save the animation file.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig, ax = self._get_fig_ax()
        plt.tight_layout()
        plt.close(fig)

        def make_frame(state_index: int) -> None:
            state = states[state_index]
            self._draw(ax, state)

        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=len(states),
            interval=interval,
        )
        if save:
            self._animation.save(path)

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
            plt.axis("off")
            if not plt.isinteractive():
                fig.show()
            ax = fig.add_subplot()
        return fig, ax

    def _draw(self, ax: plt.Axes, state: State) -> None:
        ax.clear()
        ax.set_xticks(jnp.arange(-0.5, self.board_width - 1, 1))
        ax.set_yticks(jnp.arange(-0.5, self.board_height - 1, 1))
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
        for i in range(self.board_height):
            for j in range(self.board_width):
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
            # Block for 0.5 seconds.
            fig.canvas.start_event_loop(0.5)

    def _clear_display(self) -> None:
        if jumanji.environments.is_colab():
            import IPython.display

            IPython.display.clear_output(True)
