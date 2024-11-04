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

from typing import Any, Optional, Sequence, Tuple

import jax.numpy as jnp
import matplotlib.animation
import matplotlib.pyplot as plt
from matplotlib.layout_engine import TightLayoutEngine

import jumanji
import jumanji.environments
from jumanji.environments.swarms.common.viewer import draw_agents, format_plot
from jumanji.environments.swarms.predator_prey.types import State
from jumanji.viewer import Viewer


class PredatorPreyViewer(Viewer):
    def __init__(
        self,
        figure_name: str = "PredatorPrey",
        figure_size: Tuple[float, float] = (6.0, 6.0),
        predator_color: str = "red",
        prey_color: str = "green",
    ) -> None:
        """Viewer for the `PredatorPrey` environment.

        Args:
            figure_name: the window name to be used when initialising the window.
            figure_size: tuple (height, width) of the matplotlib figure window.
        """
        self._figure_name = figure_name
        self._figure_size = figure_size
        self.predator_color = predator_color
        self.prey_color = prey_color
        self._animation: Optional[matplotlib.animation.Animation] = None

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
        fig, ax = plt.subplots(
            num=f"{self._figure_name}Anim", figsize=self._figure_size
        )
        fig, ax = format_plot(fig, ax)

        predators_quiver = draw_agents(ax, states[0].predators, self.predator_color)
        prey_quiver = draw_agents(ax, states[0].prey, self.prey_color)

        def make_frame(state: State) -> Any:
            # Rather than redraw just update the quivers properties
            predators_quiver.set_offsets(state.predators.pos)
            predators_quiver.set_UVC(
                jnp.cos(state.predators.heading), jnp.sin(state.predators.heading)
            )
            prey_quiver.set_offsets(state.prey.pos)
            prey_quiver.set_UVC(
                jnp.cos(state.prey.heading), jnp.sin(state.prey.heading)
            )
            return ((predators_quiver, prey_quiver),)

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
        draw_agents(ax, state.predators, self.predator_color)
        draw_agents(ax, state.prey, self.prey_color)

    def _get_fig_ax(self) -> Tuple[plt.Figure, plt.Axes]:
        exists = plt.fignum_exists(self._figure_name)
        if exists:
            fig = plt.figure(self._figure_name)
            ax = fig.get_axes()[0]
        else:
            fig = plt.figure(self._figure_name, figsize=self._figure_size)
            fig.set_layout_engine(
                layout=TightLayoutEngine(pad=False, w_pad=0.0, h_pad=0.0)
            )
            if not plt.isinteractive():
                fig.show()
            ax = fig.add_subplot()

        fig, ax = format_plot(fig, ax)
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
