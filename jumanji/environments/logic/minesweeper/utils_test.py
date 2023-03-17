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

from itertools import product
from typing import Tuple

import pytest
from jax import numpy as jnp

from jumanji.environments.logic.minesweeper.types import State
from jumanji.environments.logic.minesweeper.utils import (
    count_adjacent_mines,
    explored_mine,
)


@pytest.mark.parametrize(
    "action, expected_explored_mine_result",
    zip(
        product(range(4), range(4)),
        [
            True,
            True,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ],
    ),
)
def test_explored_mine(
    manual_start_state: State,
    action: Tuple[int, int],
    expected_explored_mine_result: bool,
) -> None:
    """Test whether mines are being explored"""
    action_height, action_width = action
    assert (
        explored_mine(
            manual_start_state,
            jnp.array([action_height, action_width], dtype=jnp.int32),
        )
        == expected_explored_mine_result
    )


@pytest.mark.parametrize(
    "action, expected_count_adjacent_mines_result",
    zip(product(range(4), range(4)), [2, 4, 2, 2, 4, 8, 4, 3, 2, 4, 2, 2, 2, 3, 2, 1]),
)
def test_count_adjacent_mines(
    manual_start_state: State,
    action: Tuple[int, int],
    expected_count_adjacent_mines_result: int,
) -> None:
    """Test whether the mine counting function is working as expected"""
    action_height, action_width = action
    assert (
        count_adjacent_mines(
            manual_start_state,
            jnp.array([action_height, action_width], dtype=jnp.int32),
        )
        == expected_count_adjacent_mines_result
    )
