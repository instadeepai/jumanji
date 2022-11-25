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

import os

import jax.numpy as jnp
import jax.random as random
import pytest

from jumanji.environments.routing.routing.env import Routing
from jumanji.environments.routing.routing.env_viewer import RoutingViewer


class TestViewer:
    width = height = 500

    @pytest.fixture(scope="module")
    def env(self) -> Routing:
        """Creates the Routing environment."""
        env = Routing(8, 8, 2)

        return env

    @pytest.fixture(scope="module")
    def viewer(self, env: Routing) -> RoutingViewer:
        """Creates a viewer for the Routing environment."""
        return RoutingViewer(
            env.num_agents, env.rows, env.cols, self.width, self.height
        )

    def test_render(self, env: Routing, viewer: RoutingViewer) -> None:
        """Tests that the RoutingViewer.render() does not raise any errors."""
        state, timestep = env.reset(random.PRNGKey(0))
        viewer.render(state.grid)

        state, timestep = env.step(state, jnp.array([1, 2]))
        viewer.render(state.grid)
        viewer.render(timestep.observation[0])
        viewer.render(timestep.observation[1])

    def test__frame_shape(self, env: Routing, viewer: RoutingViewer) -> None:
        """Tests that the returned pixel array is of the correct shape."""
        state, timestep = env.reset(random.PRNGKey(0))

        frame = viewer.render(state.grid)

        assert frame.shape == (viewer.width, viewer.height, 3)

    def test__save_frame(self, env: Routing, viewer: RoutingViewer) -> None:
        """Tests that saving an image functions correctly."""
        state, timestep = env.reset(random.PRNGKey(0))
        viewer.render(state.grid, "saved_image.png")
        assert os.path.isfile("./saved_image.png")
        os.remove("./saved_image.png")

    def test__draw_shape(self, env: Routing, viewer: RoutingViewer) -> None:
        """Tests that a `TypeError` is thrown when RoutingViewer._draw_shape() is called with
        incorrect arguments.
        """
        viewer._draw_shape((1, 2, 3, 4), 1)

        with pytest.raises(TypeError):
            viewer._draw_shape((1, 2, 3), 1)  # type: ignore

        with pytest.raises(TypeError):
            viewer._draw_shape((1, 2, 3, 4), 1.5)  # type: ignore

        with pytest.raises(TypeError):
            viewer._draw_shape((1, 2, 3, 4, 6), 1)  # type: ignore

    def test__close(self, env: Routing, viewer: RoutingViewer) -> None:
        """Test closing the viewer does not throw an error."""
        viewer.close()
