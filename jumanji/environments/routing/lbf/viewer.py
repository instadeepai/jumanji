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

import math
from typing import Callable, Optional, Sequence, Tuple

import chex
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from numpy.typing import NDArray

import jumanji
import jumanji.environments.routing.robot_warehouse.constants as constants
from jumanji.environments.routing.lbf.types import Agent, Entity, Food, State
from jumanji.tree_utils import tree_slice
from jumanji.viewer import Viewer


class LevelBasedForagingViewer(Viewer):
    def __init__(
        self,
        grid_size: Tuple[int, int],
        name: str = "RobotWarehouse",
        render_mode: str = "human",
    ) -> None:
        """Viewer for the RobotWarehouse environment.

        Args:
            grid_size: the size of the warehouse floor grid (width, height)
            name: custom name for the Viewer. Defaults to `RobotWarehouse`.
        """
        self._name = name
        self.rows, self.cols = grid_size

        self.cell_size = 30
        self.icon_size = int(self.cell_size / 3)
        self.adjust_center = self.cell_size / 2

        self.width = 1 + self.cols * (self.cell_size + 1)
        self.height = 1 + self.rows * (self.cell_size + 1)

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
        """Render the given state of the `RobotWarehouse` environment.

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
        fig = plt.figure(f"{self._name}Animation", figsize=constants._FIGURE_SIZE)
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
        fig = plt.figure(self._name, figsize=constants._FIGURE_SIZE, facecolor="black")
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
        self._draw_agents(state.agents, ax)
        self._draw_foods(state.foods, ax)

    def _entity_position(self, entity: Entity) -> Tuple[float, float]:
        """Return the position of an entity on the grid."""
        row, col = entity.position
        return (
            row * self.cell_size + 1 + row + self.adjust_center,
            col * self.cell_size + 1 + col + self.adjust_center,
        )

    def _draw_grid(self, ax: plt.Axes) -> None:
        """Draw grid of warehouse floor."""
        lines = []
        # VERTICAL LINES
        for r in range(self.rows + 1):
            lines.append(
                [
                    (0, (self.cell_size + 1) * r + 1),
                    ((self.cell_size + 1) * self.cols, (self.cell_size + 1) * r + 1),
                ]
            )

        # HORIZONTAL LINES
        for c in range(self.cols + 1):
            lines.append(
                [
                    ((self.cell_size + 1) * c + 1, 0),
                    ((self.cell_size + 1) * c + 1, (self.cell_size + 1) * self.rows),
                ]
            )

        lc = LineCollection(lines, colors=(1, 1, 1))
        ax.add_collection(lc)

    def _draw_foods(self, foods: Food, ax: plt.Axes) -> None:
        """Draw the foods on the grid."""
        num_foods = len(foods.level)

        for i in range(num_foods):
            food = tree_slice(foods, i)
            if food.eaten:
                continue

            patch = plt.Circle(
                self._entity_position(food),
                radius=self.icon_size / 1.5,
                facecolor="red",
            )
            ax.add_patch(patch)

    def _draw_agents(self, agents: Agent, ax: plt.Axes) -> None:
        """Draw the agents on the grid."""
        num_agents = len(agents.level)

        for i in range(num_agents):
            agent = tree_slice(agents, i)
            cell_center = self._entity_position(agent)
            anchor_point = (
                cell_center[0] - self.icon_size / 2,
                cell_center[1] - self.icon_size / 2,
            )
            patch = plt.Rectangle(
                anchor_point, self.icon_size, self.icon_size, facecolor="white"
            )
            ax.add_patch(patch)

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
