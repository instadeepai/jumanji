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

# flake8: noqa: CCR001

from typing import Callable, Optional, Sequence, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from numpy.typing import NDArray

import jumanji
import jumanji.environments.routing.lbf.constants as constants
from jumanji.environments.routing.lbf.types import Agent, Entity, Food, State
from jumanji.tree_utils import tree_slice
from jumanji.viewer import Viewer


class LevelBasedForagingViewer(Viewer):
    def __init__(
        self,
        grid_size: int,
        name: str = "LevelBasedForaging",
        render_mode: str = "human",
    ) -> None:
        """Viewer for the LevelBasedForaging environment.

        Args:
            grid_size: the size of the grid (width, height)
            name: custom name for the Viewer. Defaults to `LevelBasedForaging`.
        """
        self._name = name
        self.rows, self.cols = (grid_size, grid_size)
        self.grid_size = 30

        self.icon_size = self.grid_size * 5 / self.rows

        self.width = 1 + self.cols * (self.grid_size + 1)
        self.height = 1 + self.rows * (self.grid_size + 1)

        self._display: Callable[[plt.Figure], Optional[NDArray]]
        if render_mode == "rgb_array":
            self._display = self._display_rgb_array
        elif render_mode == "human":
            self._display = self._display_human
        else:
            raise ValueError(f"Invalid render mode: {render_mode}")

        # The animation must be stored in a variable that lives as long as the
        # animation should run. Otherwise, the animation will get garbage-collected.
        self._animation: Optional[animation.Animation] = None

    def render(self, state: State) -> Optional[NDArray]:
        """Render the given state of the `LevelBasedForaging` environment.

        Args:
            state: the environment state to render.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._prepare_figure(ax)
        self._draw_state(ax, state)
        return self._display(fig)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> animation.FuncAnimation:
        """Create an animation from a sequence of environment states.

        Args:
            states: sequence of environment states corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig = plt.figure(
            f"{self._name}Animation",
            figsize=constants._FIGURE_SIZE,
            facecolor=constants._GRID_COLOR,
        )
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax = fig.add_subplot(111)
        plt.close(fig)
        self._prepare_figure(ax)

        def make_frame(state: State) -> None:
            ax.clear()
            self._prepare_figure(ax)
            self._draw_state(ax, state)

        # Create the animation object.
        self._animation = animation.FuncAnimation(
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
        plt.close(self._name)

    def _clear_display(self) -> None:
        if jumanji.environments.is_colab():
            import IPython.display

            IPython.display.clear_output(True)

    def _get_fig_ax(self) -> Tuple[plt.Figure, plt.Axes]:
        recreate = not plt.fignum_exists(self._name)
        fig = plt.figure(
            self._name, figsize=constants._FIGURE_SIZE, facecolor=constants._GRID_COLOR
        )
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        if recreate:
            fig.tight_layout()
            if not plt.isinteractive():
                fig.show()
            ax = fig.add_subplot(111)
        else:
            ax = fig.get_axes()[0]
        return fig, ax

    def _prepare_figure(self, ax: plt.Axes) -> None:
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.patch.set_alpha(0.0)
        ax.set_axis_off()

        ax.set_aspect("equal", "box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    def _draw_state(self, ax: plt.Axes, state: State) -> None:
        self._draw_grid(ax)
        self._draw_food(state.food_items, ax)
        self._draw_agents(state.agents, ax)

    def _draw_grid(self, ax: plt.Axes) -> None:
        """Draw the grid."""
        lines = [
            [
                (0, (self.grid_size + 1) * r + 1),
                ((self.grid_size + 1) * self.cols, (self.grid_size + 1) * r + 1),
            ]
            for r in range(self.rows + 1)
        ]
        # HORIZONTAL LINES
        lines.extend(
            [
                ((self.grid_size + 1) * c + 1, 0),
                ((self.grid_size + 1) * c + 1, (self.grid_size + 1) * self.rows),
            ]
            for c in range(self.cols + 1)
        )
        lc = LineCollection(lines, colors=(constants._LINE_COLOR,))
        ax.add_collection(lc)

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

    def _display_rgb_array(self, fig: plt.Figure) -> NDArray:
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())

    def _draw_agents(self, agents: Agent, ax: plt.Axes) -> None:
        """Draw the agents on the grid."""
        num_agents = len(agents.level)

        for i in range(num_agents):
            agent = tree_slice(agents, i)
            cell_center = self._entity_position(agent)

            # Read the image file
            img_path = pkg_resources.resource_filename(
                "jumanji", "environments/routing/lbf/imgs/agent.png"
            )
            img = plt.imread(img_path)

            # Create an OffsetImage and add it to the axis
            imagebox = OffsetImage(img, zoom=self.icon_size / self.grid_size)
            ab = AnnotationBbox(imagebox, (cell_center[0], cell_center[1]), frameon=False, zorder=0)
            ax.add_artist(ab)

            # Add a rectangle (polygon) next to the agent with the agent's level
            self.draw_badge(agent.level, cell_center, ax)

    def _draw_food(self, food_items: Food, ax: plt.Axes) -> None:
        """Draw the food on the grid."""
        num_food = len(food_items.level)

        for i in range(num_food):
            food = tree_slice(food_items, i)
            if food.eaten:
                continue

            # Read the image file
            img_path = pkg_resources.resource_filename(
                "jumanji", "environments/routing/lbf/imgs/apple.png"
            )
            img = plt.imread(img_path)
            cell_center = self._entity_position(food)
            self.draw_badge(food.level, cell_center, ax)

            # Create an OffsetImage and add it to the axis
            imagebox = OffsetImage(img, zoom=self.icon_size / self.grid_size)
            ab = AnnotationBbox(imagebox, (cell_center[0], cell_center[1]), frameon=False, zorder=0)
            ax.add_artist(ab)

            # Add a rectangle (polygon) next to the agent with the food's level

    def _entity_position(self, entity: Entity) -> Tuple[float, float]:
        """Return the position of an entity on the grid."""
        row, col = entity.position
        row = self.rows - row - 1  # pyglet rendering is reversed
        x_center = (self.grid_size + 1) * col + self.grid_size // 2 + 1
        y_center = (self.grid_size + 1) * row + self.grid_size // 2 + 1
        return (
            x_center,
            y_center,
        )

    def draw_badge(self, level: int, anchor_point: Tuple[float, float], ax: plt.Axes) -> None:
        resolution = 6
        radius = self.grid_size / 6

        badge_center_x = anchor_point[0] + self.grid_size / 3 - 3
        badge_center_y = anchor_point[1] - self.grid_size / 3

        # make a circle
        verts = []
        for i in range(resolution):
            angle = 2 * np.pi * i / resolution

            x_radius = radius * np.cos(angle)
            x = x_radius + badge_center_x + 1

            y_radius = radius * np.sin(angle) + 1
            y = y_radius + badge_center_y
            verts += [[x, y]]

            circle = plt.Polygon(
                verts,
                edgecolor="white",
                facecolor=constants._GRID_COLOR,
            )

        ax.add_patch(circle)
        fontsize = 10 if self.rows <= 10 else (6 if 10 < self.rows < 15 else 5)
        ax.annotate(
            str(level),
            xy=(badge_center_x + 1, badge_center_y + 1),
            color="white",
            ha="center",
            va="center",
            zorder=10,
            fontsize=fontsize,
            weight="bold",
        )
