from abc import ABC, abstractmethod

import numpy as np


class AbstractBoard(ABC):
    def __init__(self, rows: int, cols: int, num_agents: int):
        self._rows = rows
        self._cols = cols
        self._num_agents = num_agents # upper limit
        self._wires_on_board = 0

    @abstractmethod
    def _is_num_agents_hit(self):
        return self.num_agents == self._wires_on_board
    @property
    def rows(self):
        return self._rows

    @rows.setter
    def rows(self, value):
        self._rows = value

    @property
    def cols(self):
        return self._cols

    @cols.setter
    def cols(self, value):
        self._cols = value

    @property
    def num_agents(self):
        return self._cols

    @num_agents.setter
    def num_agents(self, value):
        self._num_agents = value

    @property
    def wires_on_board(self):
        return self._wires_on_board

    @abstractmethod
    def return_training_board(self) -> np.ndarray:
        pass

    @abstractmethod
    def return_solved_board(self) -> np.ndarray:
        pass
