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
# limitations under the License.i

import abc

import chex
import jax.numpy as jnp
import numpy as np

from jumanji.environments.logic.sudoku.constants import BOARD_WIDTH
from jumanji.environments.logic.sudoku.types import State
from jumanji.environments.logic.sudoku.utils import update_action_mask


class Generator(abc.ABC):
    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> State:
        """Call method responsible for generating a new state.

        Args:
            key: jax random key for any stochasticity used in the instance
                generation process.

        Returns:
            A Sudoku State.
        """
        pass


# class CSVGenerator(Generator):
#     def __init__(self, data_path: Path):
#         self.data_path = data_path
#         board = np.loadtxt(self.data_path, delimiter=",")
#         action_mask = np.ma.make_mask(board)
#         board = jnp.array(board, dtype=jnp.int8) - 1
#         action_mask = jnp.array(action_mask, dtype=jnp.int8)
#         action_mask = 1 - jnp.expand_dims(action_mask, -1).repeat(BOARD_WIDTH, axis=-1)
#         action_mask = update_action_mask(action_mask, board)

#         self.state = State(board=board, action_mask=action_mask)

#     def __call__(self, key: PRNGKey):
#         return self.state


class DummyGenerator(Generator):
    def __init__(
        self,
    ):
        board = jnp.array(
            [
                [7, 8, 0, 4, 0, 0, 1, 2, 0],
                [6, 0, 0, 0, 7, 5, 0, 0, 9],
                [0, 0, 0, 6, 0, 1, 0, 7, 8],
                [0, 0, 7, 0, 4, 0, 2, 6, 0],
                [0, 0, 1, 0, 5, 0, 9, 3, 0],
                [9, 0, 4, 0, 6, 0, 0, 0, 5],
                [0, 7, 0, 3, 0, 0, 0, 1, 2],
                [1, 2, 0, 0, 0, 7, 4, 0, 0],
                [0, 4, 9, 2, 0, 6, 0, 0, 7],
            ]
        )

        action_mask = np.ma.make_mask(board)
        board = jnp.array(board, dtype=jnp.int32) - 1
        action_mask = jnp.array(action_mask, dtype=jnp.int32)
        action_mask = 1 - jnp.expand_dims(action_mask, -1).repeat(BOARD_WIDTH, axis=-1)
        action_mask = update_action_mask(action_mask, board).astype(bool)
        self._board = board
        self._action_mask = action_mask

    def __call__(self, key: chex.PRNGKey):
        return State(board=self._board, action_mask=self._action_mask, key=key)
