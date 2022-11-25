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

from jumanji.environments.routing.snake.types import Position
from jumanji.environments.routing.snake.utils import position_from_coordinates


def test_position_from_coordinates() -> None:
    """Validates position_from_coordinates method from utils.
    Checks the conversion to a single Position and to an array of positions.
    """
    coordinates = jnp.array([3, 2])
    position = position_from_coordinates(coordinates)
    assert position == Position(jnp.array(3), jnp.array(2))
    coordinates = jnp.array([[3, 2], [1, 4], [0, 1]])
    position = position_from_coordinates(coordinates)
    assert position == Position(jnp.array([3, 1, 0]), jnp.array([2, 4, 1]))
