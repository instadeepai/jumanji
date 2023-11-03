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

import jax.numpy as jnp
import matplotlib.animation
import matplotlib.pyplot as plt

import jumanji.environments
from jumanji.environments.logic.solitaire.types import State
from jumanji.viewer import Viewer


class SolitaireViewer(Viewer):
    COLORS = {
        "bg": "#964b00",
        "empty": "#321414",
        "peg": "#ff4500",
    }

    def __init__(
        self,
        name: str = "Solitaire",
        board_size: int = 7,
    ) -> None:
        """Viewer for the Solitaire environment.

        Args:
            name: the window name to be used when initialising the window.
            board_size: size of the board.
        """
        self._name = name
        self._board_size = board_size

        # The animation must be stored in a variable that lives as long as the
        # animation should run. Otherwise, the animation will get garbage-collected.
        self._animation: Optional[matplotlib.animation.Animation] = None

    def render(self, state: State) -> None:
        """Renders the current state of the board.

        Args:
            state: is the current state to be rendered.
        """
        self._clear_display()
        # Get the figure and axes for the board.
        fig, ax = self.get_fig_ax()
        # Set the figure title to display the current score.
        fig.suptitle(f"{self._name}    Remaining: {int(state.remaining)}", size=20)
        # Draw the board
        self.draw_board(ax, state)
        self._display_human(fig)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 500,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Creates an animated gif of the board based on the sequence of states.

        Args:
            states: is a list of `State` objects representing the sequence of states.
            interval: the delay between frames in milliseconds, default to 500.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation object that can be saved as a GIF, MP4, or rendered with HTML.
        """
        # Set up the figure and axes for the board.
        fig, ax = self.get_fig_ax()
        fig.suptitle(f"{self._name}    Remaining: 0", size=20)
        plt.tight_layout()

        # Define a function to animate a single state.
        def make_frame(state_index: int) -> None:
            state = states[state_index]
            self.draw_board(ax, state)
            fig.suptitle(f"{self._name}    Remaining: {int(state.remaining)}", size=20)

        # Create the animation object.
        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=len(states),
            interval=interval,
        )

        # Save the animation as a gif.
        if save_path:
            self._animation.save(save_path)

        return self._animation

    #
    def get_fig_ax(self) -> Tuple[plt.Figure, plt.Axes]:
        """This function returns a `Matplotlib` figure and axes object for displaying the board.

        Returns:
            A tuple containing the figure and axes objects.
        """
        # Check if a figure with an id of the name already exists.
        exists = plt.fignum_exists(self._name)
        if exists:
            # If it exists, get the figure and axes objects.
            fig = plt.figure(self._name)
            ax = fig.get_axes()[0]
        else:
            # If it doesn't exist, create a new figure and axes objects.
            fig = plt.figure(
                self._name,
                figsize=(6.0, 6.0),
                facecolor=self.COLORS["bg"],
            )
            plt.tight_layout()
            if not plt.isinteractive():
                fig.show()
            ax = fig.add_subplot()
        return fig, ax

    def render_point(self, peg: bool, ax: plt.Axes, row: int, col: int) -> None:
        """Renders a single peg/hole on the board.

        Args:
            peg: whether a peg is present in the location.
            ax: the axes on which to draw the point.
            row: the row index of the point on the board.
            col: the col index of the point on the board.
        """
        # Set the background color of the point based on its value.
        color = self.COLORS["peg"] if peg else self.COLORS["empty"]

        # Draw circle.
        circle = plt.Circle(
            (col, row),
            radius=0.2,
            color=color,
        )
        ax.add_patch(circle)

    def draw_board(self, ax: plt.Axes, state: State) -> None:
        """Draw the board with the current state.

        Args:
            ax: the axis to draw the board on.
            state: the current state.
        """
        ax.clear()
        # Get the board
        board = state.board
        # Flip so that up and down match the printed version.
        board = jnp.flip(board, 0)
        board_size = int(self._board_size)

        ax.set_facecolor(self.COLORS["bg"])
        ax.set_xticks(jnp.arange(-0.5, board_size + 0.5, 1))
        ax.set_yticks(jnp.arange(-0.5, board_size + 0.5, 1))
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

        # Iterate through each cell and render pegs/holes.
        midpoint = (board_size - 1) // 2
        for row in range(board_size):
            for col in range(board_size):
                if abs(row - midpoint) <= 1 or abs(col - midpoint) <= 1:
                    self.render_point(board[row, col], ax=ax, row=row, col=col)

    def close(self) -> None:
        plt.close(self._name)

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
