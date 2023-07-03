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

from typing import Callable, Optional, Sequence, Union

import matplotlib.animation
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image
from matplotlib.axes import Axes
from numpy.typing import NDArray

import jumanji.environments
from jumanji.environments.commons.maze_utils.maze_rendering import MazeViewer
from jumanji.environments.routing.pacman.types import Observation, State
from jumanji.environments.routing.pacman.utils import create_grid_image


class PacManViewer(MazeViewer):
    FIGURE_SIZE = (10.0, 10.0)

    def __init__(self, name: str, render_mode: str = "human") -> None:
        """
        Viewer for the `Pacman` environment.
        Args:
            name: the window name to be used when initialising the window.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        self._name = name
        self._render_mode = render_mode
        self._display: Callable[[plt.Figure], Optional[NDArray]]
        self._animation: Optional[matplotlib.animation.Animation] = None

        if render_mode == "rgb_array":
            self._display = self._display_rgb_array
        elif render_mode == "human":
            self._display = self._display_human
        else:
            raise ValueError(f"Invalid render mode: {render_mode}")

    def render(self, state: Union[Observation, State]) -> Optional[NDArray]:
        """Render the given state of the `Pacman` environment.
        Args:
            state: the environment state to render.
        Returns:
            RGB array if the render_mode is RenderMode.RGB_ARRAY.
        """
        self._clear_display()
        (
            fig,
            ax,
        ) = self._get_fig_ax()
        ax.clear()
        self._add_grid_image(state, ax)
        return self._display(fig)

    def animate(
        self,
        states: Sequence[Union[Observation, State]],
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
        fig, ax = plt.subplots(num=f"{self._name}Animation", figsize=self.FIGURE_SIZE)
        plt.close(fig)

        def make_frame(state_index: int) -> None:
            ax.clear()
            state = states[state_index]
            self._add_grid_image(state, ax)

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

    def _add_grid_image(
        self, state: Union[Observation, State], ax: Axes
    ) -> image.AxesImage:
        img = create_grid_image(state)
        ax.set_axis_off()
        return ax.imshow(img)

    def close(self) -> None:
        plt.close(self._name)

    def _clear_display(self) -> None:
        if jumanji.environments.is_colab():
            import IPython.display

            IPython.display.clear_output(True)

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
