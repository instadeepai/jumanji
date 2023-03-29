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

from typing import Callable, Optional, Sequence, Tuple

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import jumanji.environments
from jumanji.environments.packing.knapsack.types import State
from jumanji.viewer import Viewer


class KnapsackViewer(Viewer):
    FIGURE_SIZE = (5.0, 5.0)

    def __init__(
        self, name: str, render_mode: str = "human", total_budget: float = 2.0
    ) -> None:
        """Viewer for the `Knapsack` environment.

        Args:
            name: the window name to be used when initialising the window.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
            total_budget: the capacity of the knapsack.
        """
        self._name = name
        self._total_budget = total_budget

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

    def render(self, state: State) -> Optional[NDArray]:
        """Render the given state of the `Knapsack` environment.

        Args:
            state: the environment state to render.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        self._prepare_figure(ax)
        self._show_value_and_budget(ax, state)
        return self._display(fig)

    def _show_value_and_budget(self, ax: plt.Axes, state: State) -> None:
        # Initially, no items have been picked
        budget_used: np.ndarray = np.sum(state.weights, where=state.packed_items)
        total_value: np.ndarray = np.sum(state.values, where=state.packed_items)

        ax.set_title(
            f"Total value: {round(float(total_value), 2):.2f}. "
            f"Budget used: {round(float(budget_used), 2):.2f}/{self._total_budget}."
        )

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
        ax = fig.add_subplot(111)
        self._prepare_figure(ax)

        def make_frame(state_index: int) -> None:
            state = states[state_index]
            self._show_value_and_budget(ax, state)

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
        exists = plt.fignum_exists(self._name)
        if exists:
            fig = plt.figure(self._name)
            # ax = fig.add_subplot()
            ax = fig.get_axes()[0]
        else:
            fig = plt.figure(self._name, figsize=self.FIGURE_SIZE)
            if not plt.isinteractive():
                fig.show()
            ax = fig.add_subplot()
        return fig, ax

    def _prepare_figure(self, ax: plt.Axes) -> None:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        map_img = plt.imread("docs/img/knapsack.png")
        ax.imshow(map_img, extent=[0, 1, 0, 1])

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
