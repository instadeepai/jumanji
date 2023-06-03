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

from itertools import groupby
from typing import Callable, Optional, Sequence, Tuple

import chex
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import jumanji.environments
from jumanji.environments.routing.multi_cvrp.types import State
from jumanji.viewer import Viewer


class MultiCVRPViewer(Viewer):
    FIGURE_SIZE = (10.0, 10.0)
    NODE_COLOUR = "black"
    COLORMAP_NAME = "hsv"
    NODE_SIZE = 150
    DEPOT_SIZE = 250
    ARROW_WIDTH = 0.004

    def __init__(
        self,
        name: str,
        num_vehicles: int,
        num_customers: int,
        map_max: int,
        render_mode: str = "human",
    ) -> None:
        """Viewer for the MultiCVRP environment.

        Args:
            name: the window name to be used when initialising the window.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        self._name = name
        self._num_vehicles = num_vehicles
        self._num_customers = num_customers
        self._map_max = map_max

        # Each vehicle has a different colour
        self._cmap = matplotlib.cm.get_cmap(self.COLORMAP_NAME, self._num_vehicles + 1)

        # The animation must be stored in a variable that lives as long as the
        # animation should run. Otherwise, the animation will get garbage-collected.
        self._animation: Optional[matplotlib.animation.Animation] = None

        self._display: Callable[[plt.Figure], Optional[NDArray]]
        if render_mode == "rgb_array":
            self._display = self._display_rgb_array
        elif render_mode == "human":
            self._display = self._display_human
        else:
            raise ValueError(f"Invalid render mode: {render_mode}")

    def render(self, state: State, save_path: Optional[str] = None) -> chex.Array:
        """Render the state of the environment.

        Args:
            state: the current state of the environment to render.
            save_path: optional name to save frame as.

        Return:
            pixel RGB array
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
        ax.clear()
        self._prepare_figure(ax)
        self._add_tour(ax, state)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
        return self._display(fig)

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
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig = plt.figure(f"{self._name}Animation", figsize=self.FIGURE_SIZE)
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        ax = fig.add_subplot(111)
        plt.close(fig)
        self._prepare_figure(ax)

        def make_frame(state_index: int) -> None:
            state = states[state_index]
            self._add_tour(ax, state)

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

    def _clear_display(self) -> None:
        if jumanji.environments.is_colab():
            import IPython.display

            IPython.display.clear_output(True)

    def _get_fig_ax(self) -> Tuple[plt.Figure, plt.Axes]:
        recreate = not plt.fignum_exists(self._name)
        fig = plt.figure(self._name, figsize=self.FIGURE_SIZE)
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        if recreate:
            fig.tight_layout()
            if not plt.isinteractive():
                fig.show()
            ax = fig.add_subplot(111)
        else:
            ax = fig.get_axes()[0]
        return fig, ax

    def _prepare_figure(self, ax: plt.Axes) -> None:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        map_img = plt.imread("docs/img/city_map.jpeg")
        ax.imshow(map_img, extent=[0, 1, 0, 1])

    def _group_tour(self, tour: chex.Array) -> list:
        """Group the tour into routes that either (1) start and end at the depot, or, (2) start at
        the depot and end at the current city.

        Args:
            tour: x and y coordinates of the cities in the tour.

        Returns:
            tour_grouped: list of x and y coordinates that are grouped based on the above.
        """
        depot = tour[0]
        check_depot_fn = lambda x: (x != depot).all()
        tour_grouped = [
            np.array([depot] + list(g) + [depot])
            for k, g in groupby(tour, key=check_depot_fn)
            if k
        ]
        if (tour[-1] != tour[0]).all():
            tour_grouped[-1] = tour_grouped[-1][:-1]
        return tour_grouped

    def _draw_route(self, ax: plt.Axes, coords: chex.Array, col_id: int) -> None:
        """Draw the arrows and nodes for each route in the given colour."""
        x, y = coords[:, 0], coords[:, 1]

        # Compute the difference in the x- and y-coordinates to determine the distance between
        # consecutive cities.
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        ax.quiver(
            x[:-1],
            y[:-1],
            dx,
            dy,
            scale_units="xy",
            angles="xy",
            scale=1,
            width=self.ARROW_WIDTH,
            headwidth=5,
            color=self._cmap(col_id),
        )
        ax.scatter(x, y, s=self.NODE_SIZE, color=self._cmap(col_id))

    def _add_tour(self, ax: plt.Axes, state: State) -> None:
        """Add the customers and the depot to the plot, and draw each route in the tour in a different
        colour. The tour is the entire trajectory between the visited customers and a route is a
        trajectory either starting and ending at the depot or starting at the depot and ending at
        the current city."""
        x_coords, y_coords = (
            state.nodes.coordinates[:, 0] / self._map_max,
            state.nodes.coordinates[:, 1] / self._map_max,
        )

        # Draw the customers
        ax.scatter(x_coords[1:], y_coords[1:], s=self.NODE_SIZE, color=self.NODE_COLOUR)

        # Draw the arrows between customers
        if state.step_count > 0:
            # TODO (dries): Can we do this without a for loop?
            for i in range(len(state.order)):
                coords = (
                    state.nodes.coordinates[state.order[i, : state.step_count]]
                    / self._map_max
                )

                coords_grouped = self._group_tour(coords)

                # Draw each route in different colour
                for coords_route, _ in zip(
                    coords_grouped, np.arange(0, len(coords_grouped))
                ):
                    self._draw_route(ax, coords_route, i)

        # Draw the depot node
        ax.scatter(
            x_coords[0],
            y_coords[0],
            marker="s",
            s=self.DEPOT_SIZE,
            color=self.NODE_COLOUR,
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
            # Block for 2 seconds.
            fig.canvas.start_event_loop(2.0)

    def _display_rgb_array(self, fig: plt.Figure) -> NDArray:
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())
