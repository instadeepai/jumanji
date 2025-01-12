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
import numpy as np
from matplotlib.artist import Artist
from matplotlib.layout_engine import TightLayoutEngine

import jumanji
import jumanji.environments
from jumanji.environments.swarms.common.viewer import draw_agents, format_plot
from jumanji.environments.swarms.search_and_rescue.types import State
from jumanji.viewer import Viewer


class SearchAndRescueViewer(Viewer[State]):
    def __init__(
        self,
        figure_name: str = "SearchAndRescue",
        figure_size: Tuple[float, float] = (6.0, 6.0),
        searcher_color: str = "blue",
        target_found_color: str = "green",
        target_lost_color: str = "red",
        env_size: Tuple[float, float] = (1.0, 1.0),
    ) -> None:
        """Viewer for the `SearchAndRescue` environment.

        Args:
            figure_name: The window name to be used when initialising the window.
            figure_size: Tuple (height, width) of the matplotlib figure window.
            searcher_color: Color of searcher agent markers (arrows).
            target_found_color: Color of target markers when they have been found.
            target_lost_color: Color of target markers when they are still to be found.
            env_size: Tuple environment spatial dimensions, used to set the plot region.
        """
        self._figure_name = figure_name
        self._figure_size = figure_size
        self.searcher_color = searcher_color
        self.target_colors = np.array([target_lost_color, target_found_color])
        self._animation: Optional[matplotlib.animation.Animation] = None
        self.env_size = env_size

    def render(self, state: State) -> None:
        """Render a frame of the environment for a given state using matplotlib.

        Args:
            state: State object containing the current dynamics of the environment.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        self._draw(ax, state)
        self._update_display(fig)

    def animate(
        self, states: Sequence[State], interval: int, save_path: Optional[str]
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
        fig, ax = plt.subplots(num=f"{self._figure_name}Anim", figsize=self._figure_size)
        fig, ax = format_plot(fig, ax, self.env_size)

        searcher_quiver = draw_agents(ax, states[0].searchers, self.searcher_color)
        target_scatter = ax.scatter(
            states[0].targets.pos[:, 0], states[0].targets.pos[:, 1], marker="o"
        )

        def make_frame(state: State) -> Tuple[Artist, Artist]:
            searcher_quiver.set_offsets(state.searchers.pos)
            searcher_quiver.set_UVC(
                jnp.cos(state.searchers.heading), jnp.sin(state.searchers.heading)
            )
            target_colors = self.target_colors[state.targets.found.astype(jnp.int32)]
            target_scatter.set_offsets(state.targets.pos)
            target_scatter.set_color(target_colors)
            return searcher_quiver, target_scatter

        matplotlib.rc("animation", html="jshtml")
        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=states,
            interval=interval,
            blit=False,
        )

        if save_path:
            self._animation.save(save_path)

        return self._animation

    def close(self) -> None:
        """Perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        plt.close(self._figure_name)

    def _draw(self, ax: plt.Axes, state: State) -> None:
        ax.clear()
        draw_agents(ax, state.searchers, self.searcher_color)
        target_colors = self.target_colors[state.targets.found.astype(jnp.int32)]
        ax.scatter(
            state.targets.pos[:, 0], state.targets.pos[:, 1], marker="o", color=target_colors
        )

    def _get_fig_ax(self) -> Tuple[plt.Figure, plt.Axes]:
        exists = plt.fignum_exists(self._figure_name)
        if exists:
            fig = plt.figure(self._figure_name)
            ax = fig.get_axes()[0]
        else:
            fig = plt.figure(self._figure_name, figsize=self._figure_size)
            fig.set_layout_engine(layout=TightLayoutEngine(pad=False, w_pad=0.0, h_pad=0.0))
            if not plt.isinteractive():
                fig.show()
            ax = fig.add_subplot()

        fig, ax = format_plot(fig, ax, self.env_size)
        return fig, ax

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
