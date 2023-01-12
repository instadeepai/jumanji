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
from jax import random

from jumanji.environments.logic.minesweeper.constants import UNEXPLORED_ID
from jumanji.environments.logic.minesweeper.env import Minesweeper
from jumanji.environments.logic.minesweeper.types import State


@pytest.fixture
def minesweeper_env() -> Minesweeper:
    """Fixture for a default minesweeper env"""
    return Minesweeper()


@pytest.fixture
def manual_start_state() -> State:
    """Fixture for a start state chosen manually to verify the impact of certain actions"""
    empty_board = jnp.full(shape=(4, 4), fill_value=UNEXPLORED_ID, dtype=jnp.int32)
    flat_mine_locations = jnp.array([0, 1, 2, 4, 6, 8, 9, 10], dtype=jnp.int32)
    key = random.PRNGKey(0)
    return State(
        board=empty_board,
        flat_mine_locations=flat_mine_locations,
        key=key,
        step_count=jnp.int32(0),
    )
