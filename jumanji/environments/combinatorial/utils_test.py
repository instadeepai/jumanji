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
from jax import numpy as jnp

from jumanji.environments.combinatorial.utils import get_coordinates_augmentations


@pytest.fixture
def coordinates() -> jnp.array:
    coordinates = jnp.array([[0.65, 0.85], [0.18, 0.06], [0.41, 0.19], [0.92, 0.27]])
    return coordinates


@pytest.fixture
def expected_augmentations(coordinates: jnp.array) -> jnp.array:
    expected_augmentations = jnp.array(
        [
            coordinates,
            jnp.array(
                [
                    [0.35, 0.85],
                    [0.82, 0.06],
                    [0.59, 0.19],
                    [0.08, 0.27],
                ]
            ),
            jnp.array(
                [
                    [0.65, 0.15],
                    [0.18, 0.94],
                    [0.41, 0.81],
                    [0.92, 0.73],
                ]
            ),
            jnp.array(
                [
                    [0.35, 0.15],
                    [0.82, 0.94],
                    [0.59, 0.81],
                    [0.08, 0.73],
                ]
            ),
            jnp.array(
                [
                    [0.85, 0.65],
                    [0.06, 0.18],
                    [0.19, 0.41],
                    [0.27, 0.92],
                ]
            ),
            jnp.array(
                [
                    [0.85, 0.35],
                    [0.06, 0.82],
                    [0.19, 0.59],
                    [0.27, 0.08],
                ]
            ),
            jnp.array(
                [
                    [0.15, 0.65],
                    [0.94, 0.18],
                    [0.81, 0.41],
                    [0.73, 0.92],
                ]
            ),
            jnp.array(
                [
                    [0.15, 0.35],
                    [0.94, 0.82],
                    [0.81, 0.59],
                    [0.73, 0.08],
                ]
            ),
        ]
    )
    return expected_augmentations


def test_utils__get_coordinates_augmentations(
    coordinates: jnp.array, expected_augmentations: jnp.array
) -> None:
    """Checks that the augmentations of a given instance problem is computed properly."""
    assert jnp.allclose(
        expected_augmentations, get_coordinates_augmentations(coordinates)
    )
