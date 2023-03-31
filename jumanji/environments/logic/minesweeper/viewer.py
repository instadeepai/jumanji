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
import jax.numpy as jnp
import matplotlib
from matplotlib import pyplot as plt

import jumanji.environments
from jumanji.environments.logic.minesweeper.constants import (
    DEFAULT_COLOR_MAPPING,
    UNEXPLORED_ID,
)
from jumanji.environments.logic.minesweeper.types import State
from jumanji.environments.logic.minesweeper.utils import explored_mine
from jumanji.viewer import Viewer


class MinesweeperViewer(Viewer[State]):
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        color_mapping: Optional[List[str]] = None,
    ):
        """
        Args:
            num_rows: number of rows, i.e. height of the board.
            num_cols: number of columns, i.e. width of the board.
            color_mapping: colors used in rendering the cells in `Minesweeper`.
                Defaults to `DEFAULT_COLOR_MAPPING`.
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.cmap = color_mapping or DEFAULT_COLOR_MAPPING
        self.figure_name = f"{num_rows}x{num_cols} Minesweeper"
        self.figure_size = (6.0, 6.0)

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
