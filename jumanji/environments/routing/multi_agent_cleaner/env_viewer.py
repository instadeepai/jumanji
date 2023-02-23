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

from typing import Optional

from numpy.typing import NDArray

from jumanji.environments.commons.maze_utils.maze_generation import Maze
from jumanji.environments.commons.maze_utils.maze_rendering import MazeViewer
from jumanji.environments.routing.multi_agent_cleaner.constants import (
    CLEAN,
    DIRTY,
    WALL,
)
from jumanji.environments.routing.multi_agent_cleaner.types import State


class CleanerViewer(MazeViewer):
    AGENT = 3
    COLORS = {
        CLEAN: [1, 1, 1],  # White
        WALL: [0, 0, 0],  # Black
        DIRTY: [0, 1, 0],  # Green
        AGENT: [1, 0, 0],  # Red
    }

    def __init__(self, name: str, render_mode: str = "human") -> None:
        """
        Viewer for the Cleaner environment.

        Args:
            name: the window name to be used when initialising the window.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        super().__init__(name, render_mode)

    def render(self, state: State) -> Optional[NDArray]:
        """Render the given state of the `Cleaner` environment.

        Args:
            state: the environment state to render.
        """
        maze = self._overlay_agents_on_grid(state)
        return super().render(maze)

    def _overlay_agents_on_grid(self, state: State) -> Maze:
        agents_locations = state.agents_locations
        return state.grid.at[agents_locations[:, 0], agents_locations[:, 1]].set(
            self.AGENT
        )
