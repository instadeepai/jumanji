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

import matplotlib.animation
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np

import jumanji.environments
from jumanji.environments.packing.job_shop.types import State
from jumanji.viewer import Viewer


class JobShopViewer(Viewer):
    FIGURE_SIZE = (15.0, 10.0)
    COLORMAP_NAME = "hsv"

    def __init__(
        self,
        name: str,
        num_jobs: int,
        num_machines: int,
        max_num_ops: int,
        max_op_duration: int,
    ) -> None:
        """Viewer for the `JobShop` environment.

        Args:
            name: the window name to be used when initialising the window.
            num_jobs: the number of jobs that need to be scheduled.
            num_machines: the number of machines that the jobs can be scheduled on.
            max_num_ops: the maximum number of operations for any given job.
            max_op_duration: the maximum processing time of any given operation.
        """
        self._name = name
        self._num_jobs = num_jobs
        self._num_machines = num_machines
        self._max_num_ops = max_num_ops
        self._max_op_duration = max_op_duration

        # Have additional color to avoid two jobs having same color when using hsv colormap
        self._cmap = matplotlib.cm.get_cmap(self.COLORMAP_NAME, self._num_jobs + 1)

        # The animation must be stored in a variable that lives as long as the
        # animation should run. Otherwise, the animation will get garbage-collected.
        self._animation: Optional[matplotlib.animation.Animation] = None

    def render(self, state: State) -> None:
        """Render the given state of the `JobShop` environment.

        Args:
            state: the environment state to render.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        ax.set_title(f"Scheduled Jobs at Time={state.step_count}")
        ax.axvline(state.step_count, ls="--", color="red", lw=0.5)
        self._prepare_figure(ax)
        self._add_scheduled_ops(ax, state)
        return self._display_human(fig)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of states.

        Args:
            states: sequence of environment states corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation object that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig = plt.figure(f"{self._name}Animation", figsize=self.FIGURE_SIZE)
        ax = fig.add_subplot(111)
        self._prepare_figure(ax)

        def make_frame(state_index: int) -> None:
            ax.clear()
            self._prepare_figure(ax)
            state = states[state_index]
            ax.set_title(rf"Scheduled Jobs at Time={state.step_count}")
            self._add_scheduled_ops(ax, state)

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

    def _prepare_figure(self, ax: plt.Axes) -> None:
        ax.set_xlabel("Time")
        ax.set_ylabel("Machine ID")
        xlim = (
            self._num_jobs
            * self._max_num_ops
            * self._max_op_duration
            // self._num_machines
        )
        ax.set_xlim(0, xlim)
        ax.set_ylim(-0.9, self._num_machines)
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.yaxis.get_major_locator().set_params(integer=True)
        major_ticks = np.arange(0, xlim, 10)
        minor_ticks = np.arange(0, xlim, 1)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.grid(axis="x", linewidth=0.25)
        ax.set_axisbelow(True)

    def close(self) -> None:
        plt.close(self._name)

    def _get_fig_ax(self) -> Tuple[plt.Figure, plt.Axes]:
        recreate = not plt.fignum_exists(self._name)
        fig = plt.figure(self._name, figsize=self.FIGURE_SIZE)
        if recreate:
            fig.tight_layout()
            if not plt.isinteractive():
                fig.show()
            ax = fig.add_subplot(111)
        else:
            ax = fig.get_axes()[0]
        return fig, ax

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

    def _clear_display(self) -> None:
        if jumanji.environments.is_colab():
            import IPython.display

            IPython.display.clear_output(True)

    def _add_scheduled_ops(self, ax: plt.Axes, state: State) -> None:
        """Add the scheduled operations to the plot."""
        for job_id in range(self._num_jobs):
            for op_id in range(self._max_num_ops):
                start_time = state.scheduled_times[job_id, op_id]
                machine_id = state.ops_machine_ids[job_id, op_id]
                duration = state.ops_durations[job_id, op_id]
                colour = self._cmap(job_id)
                line_height = 0.8
                if start_time >= 0:
                    rectangle = matplotlib.patches.Rectangle(
                        (start_time, machine_id - line_height / 2),
                        width=duration,
                        height=line_height,
                        linewidth=1,
                        facecolor=colour,
                        edgecolor="black",
                    )
                    ax.add_patch(rectangle)

                    # Annotate the operation with the job id
                    rx, ry = rectangle.get_xy()
                    cx = rx + rectangle.get_width() / 2.0
                    cy = ry + rectangle.get_height() / 2.0
                    ax.annotate(
                        f"J{job_id}",
                        (cx, cy),
                        color="black",
                        fontsize=10,
                        ha="center",
                        va="center",
                    )
