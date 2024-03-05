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

import pytest

from jumanji.environments.logic.sliding_tile_puzzle import SlidingTilePuzzle
from jumanji.environments.logic.sliding_tile_puzzle.generator import RandomGenerator


@pytest.fixture
def sliding_tile_puzzle() -> SlidingTilePuzzle:
    """Instantiates a default SlidingTilePuzzle environment."""
    generator = RandomGenerator(grid_size=2)
    return SlidingTilePuzzle(generator=generator)
