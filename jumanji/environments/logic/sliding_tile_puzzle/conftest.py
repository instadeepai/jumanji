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

import jax
import jax.numpy as jnp
import pytest

from jumanji.environments.logic.sliding_tile_puzzle import SlidingTilePuzzle
from jumanji.environments.logic.sliding_tile_puzzle.generator import RandomWalkGenerator
from jumanji.environments.logic.sliding_tile_puzzle.types import State


@pytest.fixture
def sliding_tile_puzzle() -> SlidingTilePuzzle:
    """Instantiates a default SlidingTilePuzzle environment."""
    generator = RandomWalkGenerator(grid_size=3)
    return SlidingTilePuzzle(generator=generator)


@pytest.fixture
def state() -> State:
    key = jax.random.PRNGKey(0)
    empty_pos = jnp.array([0, 0])
    puzzle = jnp.array(
        [
            [0, 1, 3],
            [4, 2, 5],
            [7, 8, 6],
        ]
    )
    return State(puzzle=puzzle, empty_tile_position=empty_pos, key=key, step_count=0)
