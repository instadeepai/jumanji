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

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import chex
import matplotlib.animation
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import jumanji.environments
from jumanji.environments.packing.flat_pack.types import State
from jumanji.viewer import Viewer


class FlatPackViewer(Viewer):
    FIGURE_SIZE = (10, 10)

    def __init__(self, name: str, num_blocks: int, render_mode: str = "human") -> None:
        """Viewer for a `FlatPack` environment.

        Args:
            name: the window name to be used when initialising the window.
            num_blocks: number of blocks in the environment.
            render_mode: return a numpy array frame representing the environment.
        """
        self._name = name

        # Pick display method
        self._display: Callable[[plt.Figure], Optional[NDArray]]
        if render_mode == "rgb_array":
            self._display = self._display_rgb_array
        elif render_mode == "human":
            self._display = self._display_human
        else:
            raise ValueError(f"Invalid render mode: {render_mode}")

        # Create a color for each block.
        colormap_indices = np.arange(0, 1, 1 / num_blocks)
        colormap = matplotlib.cm.get_cmap("hsv", num_blocks + 1)

        self.colors = [(1.0, 1.0, 1.0, 1.0)]  # Empty grid colour should be white.
        for colormap_idx in colormap_indices:
            # Give the blocks an alpha of 0.7.
            r, g, b, _ = colormap(colormap_idx)
            self.colors.append((r, g, b, 0.7))

        # The animation must be stored in a variable that lives as long as the
        # animation should run. Otherwise, the animation will get garbage-collected.
        self._animation: Optional[matplotlib.animation.Animation] = None

    def render(self, state: State) -> Optional[NDArray]:
        """Render a FlatPack environment state.

        Args:
            state: the flat_pack environment state to be rendered.

        Returns:
            RGB array if the render_mode is RenderMode.RGB_ARRAY.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._add_grid_image(state.grid, ax)
        return self._display(fig)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of FlatPack states.

        Args:
            states: sequence of FlatPack states corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig, ax = plt.subplots(
            num=f"{self._name}Animation", figsize=FlatPackViewer.FIGURE_SIZE
        )
        plt.close(fig)

        def make_frame(state_index: int) -> None:
            ax.clear()
            state = states[state_index]
            self._add_grid_image(state.grid, ax)

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

    def close(self) -> None:
        plt.close(self._name)

    def _display_human(self, fig: plt.Figure) -> None:
        if plt.isinteractive():
            # Required to update render when using Jupyter Notebook.
            fig.canvas.draw()
            if jumanji.environments.is_notebook():
                plt.show(self._name)
        else:
            # Required to update render when not using Jupyter Notebook.
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    def _display_rgb_array(self, fig: plt.Figure) -> NDArray:
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())

    def _clear_display(self) -> None:
        if jumanji.environments.is_notebook():
            import IPython.display

            IPython.display.clear_output(True)

    def _get_fig_ax(self) -> Tuple[plt.Figure, plt.Axes]:
        recreate = not plt.fignum_exists(self._name)
        fig = plt.figure(self._name, FlatPackViewer.FIGURE_SIZE)
        if recreate:
            if not plt.isinteractive():
                fig.show()
            ax = fig.add_subplot()
        else:
            ax = fig.get_axes()[0]
        return fig, ax

    def _add_grid_image(self, grid: chex.Array, ax: plt.Axes) -> None:
        self._draw_grid(grid, ax)
        ax.set_axis_off()
        ax.set_aspect(1)
        ax.relim()
        ax.autoscale_view()

    def _draw_grid(self, grid: chex.Array, ax: plt.Axes) -> None:
        # Flip the grid upside down to match the coordinate system of matplotlib.
        grid = np.flipud(grid)
        rows, cols = grid.shape

        for row in range(rows):
            for col in range(cols):
                self._draw_grid_cell(grid[row, col], row, col, ax)

    def _draw_grid_cell(
        self, cell_value: int, row: int, col: int, ax: plt.Axes
    ) -> None:
        cell = plt.Rectangle((col, row), 1, 1, **self._get_cell_attributes(cell_value))
        ax.add_patch(cell)
        if cell_value != 0:
            ax.text(
                col + 0.5,
                row + 0.5,
                str(int(cell_value)),
                color="#606060",
                ha="center",
                va="center",
                fontsize="xx-large",
            )

    def _get_cell_attributes(self, cell_value: int) -> Dict[str, Any]:
        color = self.colors[int(cell_value)]
        return {"facecolor": color, "edgecolor": "black", "linewidth": 1}
