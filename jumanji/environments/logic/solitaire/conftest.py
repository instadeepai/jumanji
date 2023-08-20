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

import jax.numpy as jnp
import pytest

from jumanji.environments.logic.solitaire.types import Board
from jumanji.environments.logic.solitaire.utils import playing_board


@pytest.fixture
def starting_board5x5() -> Board:
    """Starting board 5x5."""
    board = playing_board(5)
    # Remove middle peg
    board = board.at[2, 2].set(False)
    return board


@pytest.fixture
def board7x7() -> Board:
    """7x7 board with limited moves."""
    board = jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    return board
