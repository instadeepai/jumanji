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


from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pygame
from chex import Array

# from jumanji.environments.combinatorial.routing.constants import HEAD, TARGET
TARGET = 0
HEAD = 2


class RoutingViewer:
    """Viewer class for the Routing environment."""

    def __init__(
        self,
        num_agents: int,
        grid_rows: int,
        grid_cols: int,
        viewer_width: int,
        viewer_height: int,
        grid_unit: int = 20,
    ) -> None:
        """
        Create a RoutingViewer instance for rendering the Routing.

        Args:
            num_agents: Number of agents in the environment.
            grid_rows: Number of rows in the grid.
            grid_cols: Number of cols in the grid.
            viewer_width: Width of the viewer in pixels.
            viewer_height: Height of the viewer in pixels.
            grid_unit: the size of the grid squares in pixels.
        """
        pygame.init()

        self.width = viewer_width
        self.height = viewer_height

        # Change for Ole to be able to run on ssh
        self.screen = pygame.display.set_mode((viewer_width, viewer_height))
        self.grid_unit = grid_unit
        self.xoff = (viewer_width - self.grid_unit * grid_cols) // 2
        self.yoff = (viewer_height - self.grid_unit * grid_rows) // 2

        rnd = np.random.RandomState()
        rnd.seed(0)
        self.palette: List[Tuple[int, int, int]] = [
            (255, 255, 255),
            (255, 0, 0),
        ]

        for _ in range(num_agents):
            r, g, b = map(int, rnd.randint(0, 192, 3))
            self.palette.append((r, g, b))

    def render(self, grid: Array, save_img: Optional[str] = None) -> Array:
        """ Render the grid of the environment.

        Args:
            grid: the grid representing the Routing instance to render.
            save_img: optional name to save frame as.

        Return:
            pixel RGB array
        """
        self.screen.fill((255, 255, 255))
        rows, cols = grid.shape

        for row in range(rows):
            for col in range(cols):
                rect = (
                    self.xoff + col * self.grid_unit, # x coordinate
                    self.yoff + row * self.grid_unit, # y coordinate
                    self.grid_unit, # x grid unit size
                    self.grid_unit, # y grid unit size
                )
                value = grid[row, col]
                self._draw_shape(rect, value)

        pygame.display.update()
        if save_img:
            pygame.image.save(self.screen, save_img)

        return jnp.array(pygame.surfarray.pixels3d(self.screen))

    def _draw_shape(self, rect: Tuple[int, int, int, int], value: int) -> None:
        """
        Draw shape in the given rectangle using the given color value.

        Args:
            rect: Rectangle to draw shape in.
            value: Color value.
        """
        color = self.palette[value if value < 2 else 2 + (value - 2) // 3]
        if value > 1 and (value - TARGET) % 3 == 0:                         # I am changing target and head to be of the new version 
            pygame.draw.ellipse(self.screen, color, rect, width=5)
        else:
            pygame.draw.rect(self.screen, color, rect)
            if value > 1 and (value - HEAD) % 3 == 0:
                pygame.draw.rect(
                    self.screen,
                    (255, 255, 255),
                    (rect[0] + 5, rect[1] + 5, rect[2] - 10, rect[3] - 10),
                )
        pygame.draw.rect(self.screen, (128, 128, 128), rect, 1)

    def close(self) -> None:
        pygame.quit()
