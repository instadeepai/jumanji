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

import chex
import jax.numpy as jnp
import pytest

from jumanji.environments.logic.rubiks_cube.constants import Face
from jumanji.environments.logic.rubiks_cube.env import RubiksCube
from jumanji.environments.logic.rubiks_cube.generator import ScramblingGenerator
from jumanji.environments.logic.rubiks_cube.utils import make_solved_cube


@pytest.fixture
def solved_cube() -> chex.Array:
    """Fixture for a solved cube"""
    return make_solved_cube(cube_size=3)


@pytest.fixture
def differently_stickered_cube() -> chex.Array:
    """Fixture for a cube with all different stickers"""
    return jnp.arange(len(Face) * 3 * 3, dtype=jnp.int8).reshape(len(Face), 3, 3)


@pytest.fixture
def expected_scramble_result() -> chex.Array:
    """Fixture for the expected result for the manual scramble used in testing"""
    return jnp.array(
        [
            [
                [Face.LEFT.value, Face.RIGHT.value, Face.DOWN.value],
                [Face.RIGHT.value, Face.UP.value, Face.RIGHT.value],
                [Face.UP.value, Face.DOWN.value, Face.FRONT.value],
            ],
            [
                [Face.LEFT.value, Face.LEFT.value, Face.UP.value],
                [Face.BACK.value, Face.FRONT.value, Face.DOWN.value],
                [Face.UP.value, Face.UP.value, Face.LEFT.value],
            ],
            [
                [Face.LEFT.value, Face.BACK.value, Face.BACK.value],
                [Face.BACK.value, Face.RIGHT.value, Face.BACK.value],
                [Face.DOWN.value, Face.DOWN.value, Face.RIGHT.value],
            ],
            [
                [Face.RIGHT.value, Face.DOWN.value, Face.DOWN.value],
                [Face.UP.value, Face.BACK.value, Face.RIGHT.value],
                [Face.FRONT.value, Face.UP.value, Face.DOWN.value],
            ],
            [
                [Face.FRONT.value, Face.FRONT.value, Face.BACK.value],
                [Face.UP.value, Face.LEFT.value, Face.LEFT.value],
                [Face.FRONT.value, Face.FRONT.value, Face.RIGHT.value],
            ],
            [
                [Face.BACK.value, Face.FRONT.value, Face.BACK.value],
                [Face.LEFT.value, Face.DOWN.value, Face.FRONT.value],
                [Face.RIGHT.value, Face.LEFT.value, Face.UP.value],
            ],
        ],
        dtype=jnp.int8,
    )


@pytest.fixture
def rubiks_cube() -> RubiksCube:
    """Instantiates a `RubiksCube` environment with 10 scrambles on reset."""
    return RubiksCube(generator=ScramblingGenerator(cube_size=3, num_scrambles_on_reset=10))
