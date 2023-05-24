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

from typing import Any, List, Optional, Sequence, Tuple

import matplotlib.animation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

import jumanji
import jumanji.environments
from jumanji.environments.routing.snake.types import State
from jumanji.viewer import Viewer


class SnakeViewer(Viewer):
    def __init__(
        self, figure_name: str = "Snake", figure_size: Tuple[float, float] = (6.0, 6.0)
    ) -> None:
        """Viewer for the `Snake` environment.

        Args:
            figure_name: the window name to be used when initialising the window.
            figure_size: tuple (height, width) of the matplotlib figure window.
        """
        self._figure_name = figure_name
        self._figure_size = figure_size

        # The animation must be stored in a variable that lives as long as the
        # animation should run. Otherwise, the animation will get garbage-collected.
        self._animation: Optional[matplotlib.animation.Animation] = None

    def render(self, state: State) -> None:
        """Render frames of the environment for a given state using matplotlib.

        Args:
            state: State object containing the current dynamics of the environment.

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
        """Create an animation from a sequence of states.

        Args:
            states: sequence of `State` corresponding to subsequent timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation object that can be saved as a GIF, MP4, or rendered with HTML.
        """
        if not states:
            raise ValueError(f"The states argument has to be non-empty, got {states}.")
        fig, ax = plt.subplots(
            num=f"{self._figure_name}Anim", figsize=self._figure_size
        )
        self._draw_board(ax, states[0])
        plt.close(fig)

        patches: List[matplotlib.patches.Patch] = []

        def make_frame(state: State) -> Any:
            while patches:
                patches.pop().remove()
            patches.extend(self._create_entities(state))
            for patch in patches:
                ax.add_patch(patch)

        # Create the animation object.
        matplotlib.rc("animation", html="jshtml")
        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=states,
            interval=interval,
        )

        # Save the animation as a gif.
        if save_path:
            self._animation.save(save_path)

        return self._animation

    def close(self) -> None:
        """Perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        plt.close(self._figure_name)

    def _get_fig_ax(self) -> Tuple[plt.Figure, plt.Axes]:
        exists = plt.fignum_exists(self._figure_name)
        if exists:
            fig = plt.figure(self._figure_name)
            ax = fig.get_axes()[0]
        else:
            fig = plt.figure(self._figure_name, figsize=self._figure_size)
            fig.set_tight_layout({"pad": False, "w_pad": 0.0, "h_pad": 0.0})
            if not plt.isinteractive():
                fig.show()
            ax = fig.add_subplot()
        return fig, ax

    def _draw(self, ax: plt.Axes, state: State) -> None:
        ax.clear()
        self._draw_board(ax, state)
        for patch in self._create_entities(state):
            ax.add_patch(patch)

    def _draw_board(self, ax: plt.Axes, state: State) -> None:
        num_rows, num_cols = state.body_state.shape[-2:]
        # Draw the square box that delimits the board.
        ax.axis("off")
        ax.plot([0, 0], [0, num_rows], "-k", lw=2)
        ax.plot([0, num_cols], [num_rows, num_rows], "-k", lw=2)
        ax.plot([num_cols, num_cols], [num_rows, 0], "-k", lw=2)
        ax.plot([num_cols, 0], [0, 0], "-k", lw=2)

    def _create_entities(self, state: State) -> List[matplotlib.patches.Patch]:
        """Loop over the different cells and draws corresponding shapes in the ax object."""
        num_rows, num_cols = state.body_state.shape[-2:]

        patches = []
        linewidth = (
            min(n * size for n, size in zip((num_rows, num_cols), self._figure_size))
            / 44.0
        )
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "", ["yellowgreen", "forestgreen"]
        )
        for row in range(num_rows):
            for col in range(num_cols):
                if state.body_state[row, col]:
                    body_cell_patch = Rectangle(
                        (col, num_rows - 1 - row),
                        1,
                        1,
                        edgecolor=cmap(1),
                        facecolor=cmap(state.body_state[row, col] / state.length),
                        fill=True,
                        lw=linewidth,
                    )
                    patches.append(body_cell_patch)
        head_patch = Circle(
            (
                state.head_position[1] + 0.5,
                num_rows - 1 - state.head_position[0] + 0.5,
            ),
            0.3,
            edgecolor=cmap(0.5),
            facecolor=cmap(0),
            fill=True,
            lw=linewidth,
        )
        patches.append(head_patch)
        fruit_patch = Circle(
            (
                state.fruit_position[1] + 0.5,
                num_rows - 1 - state.fruit_position[0] + 0.5,
            ),
            0.2,
            edgecolor="brown",
            facecolor="lightcoral",
            fill=True,
            lw=linewidth,
        )
        patches.append(fruit_patch)
        return patches

    def _update_display(self, fig: plt.Figure) -> None:
        if plt.isinteractive():
            # Required to update render when using Jupyter Notebook.
            fig.canvas.draw()
            if jumanji.environments.is_colab():
                plt.show(self._figure_name)
        else:
            # Required to update render when not using Jupyter Notebook.
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    def _clear_display(self) -> None:
        if jumanji.environments.is_colab():
            import IPython.display

            IPython.display.clear_output(True)
