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

import matplotlib
import matplotlib.pyplot as plt

import jumanji
from jumanji.environments.logic.sudoku.constants import BOARD_WIDTH
from jumanji.environments.logic.sudoku.env import State
from jumanji.viewer import Viewer


class SudokuViewer(Viewer[State]):
    def __init__(
        self,
        name: str = "Sudoku",
    ) -> None:
        self._name = name
        self._animation: Optional[matplotlib.animation.Animation] = None

    def render(
        self,
        state: State,
        save_path: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
    ) -> None:
        self._clear_display()

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            plt.title(f"{self._name}")
        else:
            fig = ax.figure
            ax.clear()
        self._draw(ax, state)
        self._display_human(fig)

    def _draw_board(self, ax: plt.Axes) -> None:
        # Draw the square box that delimits the board.
        ax.axis("off")

        _linewidth = 2.5
        ax.plot([0, 0], [0, BOARD_WIDTH], "-k", lw=_linewidth)
        ax.plot([0, BOARD_WIDTH], [BOARD_WIDTH, BOARD_WIDTH], "-k", lw=_linewidth)
        ax.plot([BOARD_WIDTH, BOARD_WIDTH], [BOARD_WIDTH, 0], "-k", lw=_linewidth)
        ax.plot([BOARD_WIDTH, 0], [0, 0], "-k", lw=_linewidth)
        for i in range(1, BOARD_WIDTH):
            if i % int(BOARD_WIDTH**0.5) == 0:
                _linewidth = 2.5
            else:
                _linewidth = 1

            hline = matplotlib.lines.Line2D(
                [0, BOARD_WIDTH], [i, i], color="k", linewidth=_linewidth
            )
            vline = matplotlib.lines.Line2D(
                [i, i], [0, BOARD_WIDTH], color="k", linewidth=_linewidth
            )
            ax.add_line(hline)
            ax.add_line(vline)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 500,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.title(f"{self._name}")

        def make_frame(state_index: int) -> None:
            state = states[state_index]
            self.render(state, ax=ax)

        animation = matplotlib.animation.FuncAnimation(
            fig, make_frame, frames=len(states), interval=interval, blit=False
        )

        if save_path:
            animation.save(save_path)

        return animation

    def _get_fig_ax(self) -> Tuple[plt.Figure, plt.Axes]:
        exists = plt.fignum_exists(self._name)
        if exists:
            fig = plt.figure(self._name)
            ax = fig.get_axes()[0]
        else:
            fig = plt.figure(
                self._name,
                figsize=(6, 6),
            )
            fig.set_tight_layout({"pad": False, "w_pad": 0.0, "h_pad": 0.0})
            if not plt.isinteractive():
                fig.show()
            ax = fig.add_subplot()
        return fig, ax

    def close(self) -> None:
        plt.close(self._name)

    def _draw(self, ax: plt.Axes, state: State) -> None:
        ax.clear()
        self._draw_board(ax)
        self._draw_figures(ax, state)

    def _draw_figures(self, ax: plt.Axes, state: State) -> None:
        """Loop over the different cells and draws corresponding shapes in the ax object."""
        board = state.board
        board_shape = board.shape

        for i in range(board_shape[0]):
            for j in range(board_shape[1]):
                x_pos = j + 0.5
                y_pos = board_shape[0] - i - 0.5
                element = board[i, j]
                if element != -1:
                    ax.text(
                        x_pos,
                        y_pos,
                        element + 1,
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=16,
                    )

    def _display_human(self, fig: plt.Figure) -> None:
        if plt.isinteractive():
            # Required to update render when using Jupyter Notebook.
            fig.canvas.draw()
            if jumanji.environments.is_colab():
                plt.show(self._name)
        else:
            # Required to update render when not using Jupyter Notebook.
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    def _clear_display(self) -> None:
        if jumanji.environments.is_colab():
            import IPython.display

            IPython.display.clear_output(True)
