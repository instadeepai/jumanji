from abc import ABC, abstractmethod

import numpy as np


class AbstractBoard(ABC):
    def __init__(self, rows: int, cols: int, num_agents: int):
        self.rows = rows
        self.cols = cols
        self.num_agents = num_agents

    @abstractmethod
    def return_training_board(self) -> np.ndarray:
        pass
