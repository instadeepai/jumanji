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

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import chex
import jax.numpy as jnp
import matplotlib.animation
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import jumanji.environments
from jumanji.environments.packing.tetris.types import State
from jumanji.viewer import Viewer


class TetrisViewer(Viewer):
    FIGURE_SIZE = (6.0, 10.0)

    def __init__(
        self, num_rows: int, num_cols: int, render_mode: str = "human"
    ) -> None:
        """
        Viewer for a `Tetris` environment.

        Args:
            name: the window name to be used when initialising the window.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._name = f"{num_rows}x{num_cols} Tetris"
        self.n_colors = 10
        self.figure_size = (6.0, 6.0)
        # Pick display method.
        self._display: Callable[[plt.Figure], Optional[NDArray]]
        if render_mode == "rgb_array":
            self._display = self._display_rgb_array
        elif render_mode == "human":
            self._display = self._display_human
        else:
            raise ValueError(f"Invalid render mode: {render_mode}")

        # Pick colors.
        colormap_indecies = jnp.arange(0, 1, 1 / self.n_colors)
        colormap = matplotlib.cm.get_cmap("hsv", self.n_colors + 1)

        self.colors = [(1.0, 1.0, 1.0, 1.0)]  # Initial color must be white.
        for colormap_idx in colormap_indecies:
            self.colors.append(colormap(colormap_idx))
        self.edgecolors = [(0.0, 0.0, 0.0), (0.9, 0.9, 0.9)]

        # The animation must be stored in a variable that lives as long as the
        # animation should run. Otherwise, the animation will get garbage-collected.
        self._animation: Optional[matplotlib.animation.Animation] = None

    def render(self, state: State) -> Optional[NDArray]:
        """Render Tetris.

        Args:
            grid: the grid of the Tetris environment to render.

        Returns:
            RGB array if the render_mode is RenderMode.RGB_ARRAY.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        fig.suptitle(f"Tetris    Score: {int(state.score)}", size=20)
        ax.invert_yaxis()
        grid = self._create_rendering_grid(state)
        self._add_grid_image(ax, grid)
        return self._display(fig)

    def _move_tetromino(
        self, state: State, old_padded_grid: chex.Array
    ) -> List[chex.Array]:
        """Shifts the tetromino from center to the selected position.

        Args:
            state: `State` object containing the current environment state.
            old_padded_grid: `chex.Array` containing the grid before placing the tetromino.

        Returns:
            grids: `List[NDArray]` contais a list of grids.
        """
        grids = []
        grid = old_padded_grid[: self.num_rows, : self.num_cols]
        center_position = self.num_cols - 4
        # step is 1 to move to the right and -1 to move to the left
        step = 1 if center_position < state.x_position else -1
        for xi in range(center_position, state.x_position + step, step):
            tetromino_zonne = jnp.zeros((4, state.grid_padded.shape[1]))
            tetromino_zonne = tetromino_zonne.at[0:4, xi : xi + 4].add(
                state.old_tetromino_rotated
            )
            # Delete the cols dedicated for the right padding
            tetromino_zonne = tetromino_zonne[:, : self.num_cols]
            # Stack the tetromino with grid position
            mixed_grid = jnp.vstack((tetromino_zonne, grid))
            grids.append(mixed_grid)
        return grids

    def _crush_lines(
        self, state: State, grid: chex.Array, n: int = 2
    ) -> List[chex.Array]:
        """Creates animation when a line is crushed by toggling its value.

        Args:
            state: `State` object containing the current environment state.
            grid: `chex.Array` (self.num_rows+4, self.num_cols)
            n: `int`, optional, defines the number of repetitions. Defaults to 2.

        Returns:
            List[chex.Array]: Sequence of grids.
        """
        animation_list = []
        for _i in range(n):
            animation_list.append(grid)
            # `State.full_lines` is a vector of booleans of shape num_rows+3.
            full_lines = jnp.concatenate(
                [jnp.full((4,), False), state.full_lines[: self.num_rows]]
            )
            full_lines_reshaped = full_lines[:, jnp.newaxis]
            animation_list.append(
                jnp.where(~full_lines_reshaped, grid, jnp.zeros((1, grid.shape[1])))
            )
        return animation_list

    def _create_rendering_grid(self, state: State) -> chex.Array:
        """Create a grid that contains tetromino and the envirement gerid.

        Args:
            state: `State` object containing the current environment state.

        Returns:
            rendering_grid: `chex.Array` (self.num_rows+4, self.num_cols)
        """
        grid = state.grid_padded[: self.num_rows, : self.num_cols]
        tetromino = jnp.zeros((4, self.num_cols))
        center_position = self.num_cols - 4
        tetromino_color_id = state.grid_padded.max() + 1
        colored_tetromino = state.new_tetromino * tetromino_color_id
        tetromino = tetromino.at[0:4, center_position : center_position + 4].set(
            colored_tetromino
        )
        rendering_grid = jnp.vstack((tetromino, grid))
        return rendering_grid

    def _drop_tetromino(
        self, state: State, old_padded_grid: chex.Array
    ) -> List[NDArray]:
        """Creates animation while the tetromino is droping verticaly.

        Args:
            state: `State` object containing the current environment state.
            old_padded_grid: `chex.Array` containing the grid before placing the last tetromino

        Returns:
            grids: List[NDArray] contais a list of grids.
        """
        grids = []
        # `y_position` describes the position of the tetromino in the grid.
        # `y_position` may contain a value -1 if it bellongs to first tetromino.
        y_position = state.y_position if state.y_position != -1 else self.num_rows - 1
        # Stack the tetromino's rows on top of the grid.
        rendering_grid = jnp.vstack(
            (jnp.zeros((4, old_padded_grid.shape[1])), old_padded_grid)
        )
        # the animation grid contains 4 rows at the top dedicated to show the tetromino.
        for yi in range(y_position + 4 + 1):
            # Place the tetromino.
            grid = rendering_grid.at[
                yi : yi + 4, state.x_position : state.x_position + 4
            ].add(state.old_tetromino_rotated)
            # Crop the grid (delete the 3 rows and columns padding at the bottom and the right.)
            grid = grid[: self.num_rows + 4, : self.num_cols]
            grids.append(grid)
        return grids

    def animate(
        self,
        states: Sequence[State],
        interval: int = 100,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of Tetris grids.

        Args:
            grids: sequence of Tetris grids corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 100.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig, ax = plt.subplots(
            num=f"{self._name}Animation", figsize=TetrisViewer.FIGURE_SIZE
        )
        plt.close(fig)

        def make_frame(grid_index: int) -> None:
            """creates a frames for each state

            Args:
                grid_index: `int`
            """
            ax.clear()
            ax.invert_yaxis()
            fig.suptitle(f"Tetris    Score: {int(scores[grid_index])}", size=20)
            grid = grids[grid_index]

            self._add_grid_image(ax, grid)

        grids = []
        scores = []
        for state in states:
            scores.append(state.score - state.reward)
            if not state.is_reset:
                old_grid = state.grid_padded_old
                x_shift_grids = self._move_tetromino(state, old_grid)
                y_shift_grids = self._drop_tetromino(state, old_grid)
                grids.extend(x_shift_grids)
                grids.extend(y_shift_grids)
                score = state.score - state.reward
                scores.extend(
                    [score for i in range(len(x_shift_grids) + len(y_shift_grids))]
                )
                if state.full_lines.sum() > 0:
                    grids += self._crush_lines(state, grids[-1])
                    scores.extend([score for i in range(len(grids) - len(scores))])
        # Create the animation object.
        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=len(grids),
            interval=interval,
        )

        # Save the animation as a gif.
        if save_path:
            self._animation.save(save_path)

        return self._animation

    def close(self) -> None:
        plt.close(self._name)

    def _get_fig_ax(self) -> Tuple[plt.Figure, plt.Axes]:
        recreate = not plt.fignum_exists(self._name)
        fig = plt.figure(self._name, TetrisViewer.FIGURE_SIZE)
        if recreate:
            if not plt.isinteractive():
                fig.show()
            ax = fig.add_subplot()
        else:
            ax = fig.get_axes()[0]
        return fig, ax

    def _add_grid_image(self, ax: plt.Axes, grid: chex.Array) -> None:
        self._draw_grid(grid, ax)
        ax.set_axis_off()
        ax.set_aspect(1)
        ax.relim()
        ax.autoscale_view()

    def _draw_grid(self, grid: chex.Array, ax: plt.Axes) -> None:
        rows, cols = grid.shape

        for row in range(rows):
            for col in range(cols):
                self._draw_grid_cell(grid[row, col], row, col, ax)

    def _draw_grid_cell(
        self, cell_value: int, row: int, col: int, ax: plt.Axes
    ) -> None:
        is_padd = row < 4
        cell = plt.Rectangle(
            (col, row), 1, 1, **self._get_cell_attributes(cell_value, is_padd)
        )
        ax.add_patch(cell)

    def _get_cell_attributes(self, cell_value: int, is_padd: bool) -> Dict[str, Any]:
        cell_value = int(cell_value)
        color_id = (
            cell_value if cell_value == 0 else cell_value % (len(self.colors) - 1) + 1
        )

        color = self.colors[color_id]
        edge_color = self.edgecolors[is_padd]
        return {"facecolor": color, "edgecolor": edge_color, "linewidth": 1}

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
