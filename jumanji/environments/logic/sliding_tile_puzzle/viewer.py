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

# Copyright 2023 InstaDeep Ltd. All rights reserved.
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

import matplotlib.animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import jumanji.environments
from jumanji.environments.logic.sliding_tile_puzzle.types import State
from jumanji.viewer import Viewer


class SlidingTilePuzzleViewer(Viewer):
    EMPTY_TILE_COLOR = "#ccc0b3"

    def __init__(self, name: str = "SlidingTilePuzzle") -> None:
        """Viewer for the Sliding Tile Puzzle environment.

        Args:
            name: the window name to be used when initialising the window.
            grid_size: size of the puzzle.
        """
        self._name = name
        self._animation: Optional[matplotlib.animation.Animation] = None
        self._color_map = mcolors.LinearSegmentedColormap.from_list(
            "", ["white", "blue"]
        )

    def render(self, state: State) -> None:
        """Renders the current state of the game puzzle.

        Args:
            state: is the current game state to be rendered.
        """
        self._clear_display()
        fig, ax = self.get_fig_ax()
        self.draw_puzzle(ax, state)
        self._display_human(fig)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Creates an animated gif of the sliding tiles puzzle game based on the sequence of game states.

        Args:
            states: is a list of `State` objects representing the sequence of game states.
            interval: the delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation object that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig, ax = self.get_fig_ax()

        def make_frame(state_index: int) -> None:
            state = states[state_index]
            self.draw_puzzle(ax, state)

        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=len(states),
            interval=interval,
        )

        if save_path:
            self._animation.save(save_path)

        return self._animation

    def get_fig_ax(self) -> Tuple[plt.Figure, plt.Axes]:
        """This function returns a `Matplotlib` figure and axes object for displaying the game puzzle.

        Returns:
            A tuple containing the figure and axes objects.
        """
        exists = plt.fignum_exists(self._name)
        if exists:
            fig = plt.figure(self._name)
            ax = fig.get_axes()[0]
        else:
            fig = plt.figure(self._name, figsize=(6.0, 6.0))
            if not plt.isinteractive():
                fig.show()
            ax = fig.add_subplot()

        return fig, ax

    def draw_puzzle(self, ax: plt.Axes, state: State) -> None:
        """Draw the game puzzle with the current state.

        Args:
            ax: the axis to draw the puzzle on.
            state: the current state of the game.
        """
        ax.clear()
        grid_size = state.puzzle.shape[0]
        ax.set_xticks(np.arange(-0.5, grid_size - 1, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size - 1, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=2)

        # Render the puzzle
        for row in range(grid_size):
            for col in range(grid_size):
                tile_value = state.puzzle[row, col]
                if tile_value == 0:
                    # Render the empty tile
                    rect = plt.Rectangle(
                        [col - 0.5, row - 0.5], 1, 1, color=self.EMPTY_TILE_COLOR
                    )
                    ax.add_patch(rect)
                else:
                    # Render the numbered tile
                    ax.text(col, row, str(tile_value), ha="center", va="center")

        # Show the image of the puzzle.
        ax.imshow(state.puzzle, cmap=self._color_map)

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
