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

from typing import TYPE_CHECKING, NamedTuple

import chex
import jax.random

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

import jax.numpy as jnp
from chex import Array


@dataclass
class State:
    """
    problem: array of coordinates for all cities
    position: index of current city
    visited_mask: binary mask (0/1 <--> unvisited/visited)
    order: array of city indices denoting route (-1 --> not filled yet)
    num_visited: how many cities have been visited
    """

    problem: Array  # (problem_size, 2)
    position: jnp.int32
    visited_mask: Array  # (problem_size,)
    order: Array  # (problem_size,)
    num_visited: jnp.int32
    key: chex.PRNGKey = jax.random.PRNGKey(0)


class Observation(NamedTuple):
    """
    problem: array of coordinates for all cities
    start_position: index of starting city
    position: index of current city
    action_mask: binary mask (0/1 <--> legal/illegal)
    """

    problem: Array
    start_position: jnp.int32
    position: jnp.int32
    action_mask: Array
