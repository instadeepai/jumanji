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

from jumanji.environments.games.snake.types import Position


def position_from_coordinates(coordinates: jnp.ndarray) -> Position:
    """Transforms an array of coordinates into a Position.

    Args:
        coordinates: array of coordinates of shape (*, 2).

    Returns:
        Position with row and col of shape (*,).
    """
    return Position(*(x.squeeze() for x in jnp.split(coordinates, 2, axis=-1)))
