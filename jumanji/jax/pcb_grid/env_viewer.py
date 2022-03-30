import time
from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pygame
from chex import Array

from jumanji.jax.pcb_grid.constants import HEAD, TARGET


class PcbGridViewer:
    """
    Viewer class for the PCB grid environment.
    """

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
        Create a PcbGridViewer instance for rendering the PcbGridEnv.

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
            color = rnd.randint(0, 192, 3)
            self.palette.append((color[0], color[1], color[2]))

    def render(
        self, grid: Array, mode: str = "human", save_img: Optional[str] = None
    ) -> Array:
        """
        Render the environment.

        Args:
            grid: the grid representing the PcbGridEnv to render.
            mode: Render mode. Options: ['human', 'fast'].
            save_img: optional name to save frame as.

        Return:
            pixel RGB array
        """
        self.screen.fill((255, 255, 255))
        rows, cols = grid.shape

        for row in range(rows):
            for col in range(cols):
                rect = (
                    self.xoff + col * self.grid_unit,
                    self.yoff + row * self.grid_unit,
                    self.grid_unit,
                    self.grid_unit,
                )
                value = grid[row, col]
                self._draw_shape(rect, value)

        pygame.display.update()
        self.maybe_sleep(mode)
        if save_img:
            pygame.image.save(self.screen, save_img)

        return jnp.array(pygame.surfarray.pixels3d(self.screen))

    @staticmethod
    def maybe_sleep(mode: str) -> None:
        """
        Sleep function to make viewing easier if given human mode and fast if given fast mode.

        Args:
            mode: Render mode. Options: ['human', 'fast'].
        """
        if mode == "human":
            time.sleep(0.2)
        elif mode == "fast":
            pass
        else:
            raise ValueError(f"Render mode '{mode}' currently not supported")

    def _draw_shape(self, rect: Tuple[int, int, int, int], value: int) -> None:
        """
        Draw shape in the given rectangle using the given color value.

        Args:
            rect: Rectangle to draw shape in.
            value: Color value.

        """
        color = self.palette[value if value < 2 else 2 + (value - 2) // 3]
        if value > 1 and (value - TARGET) % 3 == 0:
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
