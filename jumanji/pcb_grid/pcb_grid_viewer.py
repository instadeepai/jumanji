import time
from typing import List, Tuple

import numpy as np
import pygame

from jumanji.pcb_grid.pcb_grid import HEAD, TARGET, PcbGridEnv


class PcbGridViewer:
    """
    Viewer class for the PCB grid environment.
    """

    def __init__(self, env: PcbGridEnv, width: int, height: int) -> None:
        """
        Create a PcbGridViewer instance.

        Args:
            env: Environment to view.
            width: Number of cells in width.
            height: Number of cells in height.

        """
        self.env = env

        pygame.init()

        self.screen = pygame.display.set_mode((width, height))
        self.grid_unit = 20
        self.xoff = (width - self.grid_unit * env.cols) // 2
        self.yoff = (height - self.grid_unit * env.rows) // 2

        rnd = np.random.RandomState()
        rnd.seed(0)
        self.palette: List[Tuple[int, int, int]] = [
            (255, 255, 255),
            (255, 0, 0),
        ]
        for _ in range(100):
            color = rnd.randint(0, 192, 3)
            self.palette.append((color[0], color[1], color[2]))

    def render_with_mode(self, mode: str = "human") -> None:
        """
        Render the environment with a given mode,
        which varies the speed at which the environment is rendered.

        Args:
            mode: Render mode. Options: ['human', 'fast']

        """
        self.render()

        if mode == "human":
            time.sleep(0.2)
        elif mode == "fast":
            time.sleep(0.01)
        else:
            raise ValueError(f"Render mode '{mode}' currently not supported")

    def render(self) -> None:
        """
        Render the environment.

        """
        self.screen.fill((255, 255, 255))

        for row in range(self.env.rows):
            for col in range(self.env.cols):
                rect = (
                    self.xoff + col * self.grid_unit,
                    self.yoff + row * self.grid_unit,
                    self.grid_unit,
                    self.grid_unit,
                )
                value = self.env.grid[row, col]

                self._draw_shape(rect, value)

        pygame.display.update()

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
