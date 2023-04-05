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

from typing import List, Optional, Sequence, Tuple

import jax.numpy as jnp
import matplotlib
from matplotlib import pyplot as plt

import jumanji.environments
from jumanji.environments.logic.rubiks_cube.constants import Face
from jumanji.environments.logic.rubiks_cube.types import State
from jumanji.viewer import Viewer


class RubiksCubeViewer(Viewer[State]):
    def __init__(self, sticker_colors: Optional[list], cube_size: int):
        """
        Args:
            sticker_colors: colors used in rendering the faces of the Rubik's cube.
            cube_size: size of cube to view.
        """
        self.cube_size = cube_size
        self.sticker_colors_cmap = matplotlib.colors.ListedColormap(sticker_colors)
        self.figure_name = f"{cube_size}x{cube_size}x{cube_size} Rubik's Cube"
        self.figure_size = (6.0, 6.0)

    def render(self, state: State) -> None:
        """Render frames of the environment for a given state using matplotlib.

        Args:
            state: `State` object corresponding to the new state of the environment.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        self._draw(ax, state)
        self._update_display(fig)

    def animate(
        self,
        states: Sequence[State],
        interval: int,
        save_path: Optional[str],
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
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=self.figure_size)
        fig.suptitle(self.figure_name)
        plt.tight_layout()
        ax = ax.flatten()
        plt.close(fig)

        def make_frame(state_index: int) -> None:
            state = states[state_index]
            self._draw(ax, state)

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

    def _get_fig_ax(self) -> Tuple[plt.Figure, List[plt.Axes]]:
        exists = plt.fignum_exists(self.figure_name)
        if exists:
            fig = plt.figure(self.figure_name)
            ax = fig.get_axes()
        else:
            fig, ax = plt.subplots(
                nrows=3, ncols=2, figsize=self.figure_size, num=self.figure_name
            )
            fig.suptitle(self.figure_name)
            ax = ax.flatten()
            plt.tight_layout()
            plt.axis("off")
            if not plt.isinteractive():
                fig.show()
        return fig, ax

    def _draw(self, ax: List[plt.Axes], state: State) -> None:
        i = 0
        for face in Face:
            ax[i].clear()
            ax[i].set_title(label=f"{face}")
            ax[i].set_xticks(jnp.arange(-0.5, self.cube_size - 1, 1))
            ax[i].set_yticks(jnp.arange(-0.5, self.cube_size - 1, 1))
            ax[i].tick_params(
                top=False,
                bottom=False,
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
                labeltop=False,
                labelright=False,
            )
            ax[i].imshow(
                state.cube[i],
                cmap=self.sticker_colors_cmap,
                vmin=0,
                vmax=len(Face) - 1,
            )
            ax[i].grid(color="black", linestyle="-", linewidth=2)
            i += 1

    def _update_display(self, fig: plt.Figure) -> None:
        if plt.isinteractive():
            # Required to update render when using Jupyter Notebook.
            fig.canvas.draw()
            if jumanji.environments.is_colab():
                plt.show(self.figure_name)
        else:
            # Required to update render when not using Jupyter Notebook.
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    def _clear_display(self) -> None:
        if jumanji.environments.is_colab():
            import IPython.display

            IPython.display.clear_output(True)

    def close(self) -> None:
        """Perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        plt.close(self.figure_name)
