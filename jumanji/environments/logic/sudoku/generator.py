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
import jax
import jax.numpy as jnp

from jumanji.environments.logic.sudoku.constants import (
    INITIAL_BOARD_SAMPLE,
    SOLVED_BOARD_SAMPLE,
)
from jumanji.environments.logic.sudoku.types import State
from jumanji.environments.logic.sudoku.utils import get_action_mask


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


class DummyGenerator(Generator):
    """Generates always the same board, for debugging purpose."""

    def __init__(
        self,
    ) -> None:
        board = jnp.asarray(INITIAL_BOARD_SAMPLE, dtype=jnp.int32)
        solved_board = jnp.asarray(SOLVED_BOARD_SAMPLE, dtype=jnp.int32)

        board = jnp.asarray(board, dtype=jnp.int32) - 1
        action_mask = get_action_mask(board)
        self._solved_board = jnp.asarray(solved_board, dtype=jnp.int32) - 1
        self._board = board
        self._action_mask = action_mask

    def __call__(self, key: chex.PRNGKey) -> State:
        return State(board=self._board, action_mask=self._action_mask, key=key)


class DatabaseGenerator(Generator):
    """Generates a board by sampling uniformly inside a puzzle database"""

    def __init__(self, database: chex.Array):
        """
        Args:
            database: a jnp.ndarray of shape (num_boards, 9, 9), the expected format is
            a 0 for an empty cell and and integer between 1 and 9 for a filled cell.
        """

        self._boards = jnp.asarray(database)

    def __call__(self, key: chex.PRNGKey) -> State:
        key, idx_key = jax.random.split(key)
        idx = jax.random.randint(
            idx_key, shape=(), minval=0, maxval=self._boards.shape[0]
        )
        board = self._boards.take(idx, axis=0)
        board = jnp.asarray(board, dtype=jnp.int32) - 1
        action_mask = get_action_mask(board)

        return State(board=board, action_mask=action_mask, key=key)
