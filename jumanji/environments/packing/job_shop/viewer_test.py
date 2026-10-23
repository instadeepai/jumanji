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

from io import BytesIO
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import AbstractMovieWriter
from matplotlib.figure import Figure
from PIL import Image

from jumanji.environments.packing.job_shop.generator import RandomGenerator
from jumanji.environments.packing.job_shop.types import State
from jumanji.environments.packing.job_shop.viewer import JobShopViewer


def make_state(
    scheduled_times: list[int], durations: list[int], step_count: int = 0, num_machines: int = 2
) -> State:
    """Make a single job whose operations all use machine zero."""
    state = RandomGenerator(1, num_machines, len(durations), max(durations))(jax.random.PRNGKey(0))
    return state.replace(
        ops_machine_ids=jnp.zeros((1, len(durations)), dtype=jnp.int32),
        ops_durations=jnp.array([durations]),
        scheduled_times=jnp.array([scheduled_times]),
        step_count=jnp.array(step_count),
    )


@pytest.mark.parametrize("num_machines", [1, 2, 4])
def test_render_keeps_all_operations_visible(num_machines: int) -> None:
    """A serial job can exceed the average workload per machine."""
    viewer = JobShopViewer("serial-job", 1, num_machines, 2, 5, render_mode="rgb_array")
    state = make_state([0, 5], [5, 5], step_count=6, num_machines=num_machines)
    try:
        image = viewer.render(state)
        assert image is not None and image.ndim == 3
        ax = plt.figure("serial-job").axes[0]
        assert len(ax.patches) == 2
        for operation in ax.patches:
            assert ax.get_xlim()[0] <= operation.get_x()
            assert operation.get_x() + operation.get_width() <= ax.get_xlim()[1]
    finally:
        viewer.close()


@pytest.mark.parametrize("scheduled_times", [[-1], [0]])
def test_render_small_instances_have_positive_time_range(scheduled_times: list[int]) -> None:
    """A horizon rounded down to zero must not hide a one-step operation."""
    viewer = JobShopViewer("small-job", 1, 4, 1, 1, render_mode="rgb_array")
    try:
        viewer.render(make_state(scheduled_times, [1], num_machines=4))
        assert plt.figure("small-job").axes[0].get_xlim()[1] >= 1
    finally:
        viewer.close()


def test_render_unscheduled_operations_do_not_expand_time_range() -> None:
    """Unscheduled duration values must not be treated as scheduled end times."""
    viewer = JobShopViewer("partial-job", 1, 4, 2, 50, render_mode="rgb_array")
    try:
        viewer.render(make_state([0, -1], [2, 50], num_machines=4))
        assert plt.figure("partial-job").axes[0].get_xlim() == (0, 25)
    finally:
        viewer.close()


def test_animation_keeps_a_fixed_range_covering_all_frames() -> None:
    """Animation frames must share a viewport that contains the complete schedule."""
    viewer = JobShopViewer("animated-job", 1, 2, 2, 5, render_mode="rgb_array")
    states = [make_state([-1, -1], [5, 5]), make_state([0, 5], [5, 5], step_count=6)]
    frames: list[tuple[float, float]] = []
    images: list[Image.Image] = []

    class FrameWriter(AbstractMovieWriter):
        """Capture actual animation frames without a GUI timer or external encoder."""

        def setup(self, fig: Figure, outfile: str, dpi: float | None = None) -> None:
            self.fig = fig

        def grab_frame(self, **savefig_kwargs: Any) -> None:
            self.fig.canvas.draw()
            frames.append(self.fig.axes[0].get_xlim())
            image = Image.fromarray(np.asarray(self.fig.canvas.buffer_rgba()).copy())
            images.append(image.convert("RGB"))

        def finish(self) -> None:
            pass

    try:
        animation = viewer.animate(states)
        animation.save("unused.gif", writer=FrameWriter())
        assert frames and all(limits == frames[0] for limits in frames)
        assert all(right >= 10 for _, right in frames)
        gif = BytesIO()
        images[0].save(gif, format="GIF", save_all=True, append_images=images[1:])
        gif.seek(0)
        with Image.open(gif) as image:
            assert image.n_frames == 2
        np.testing.assert_array_equal(states[-1].scheduled_times, [[0, 5]])
    finally:
        viewer.close()
