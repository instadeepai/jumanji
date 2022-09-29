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

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

import jax.numpy as jnp
from chex import Array


@dataclass
class State:
    """
    problem: array with the coordinates of all nodes (+ depot) and their cost
    position: index of the current node
    capacity: current capacity of the vehicle
    visited_mask: binary mask (0/1 <--> unvisited/visited)
    order: array of node indices denoting route (-1 --> not filled yet)
    num_total_visits: number of performed visits (it can count depot multiple times)
    """

    problem: Array  # (problem_size + 1, 3)
    position: jnp.int32
    capacity: jnp.float32
    visited_mask: Array  # (problem_size + 1,)
    order: Array  # (2 * problem_size,) - the size is worst-case (going back to depot after visiting each node)
    num_total_visits: jnp.int32


class Observation(NamedTuple):
    """
    problem: array with the coordinates of all nodes (+ depot) and their cost
    position: index of the current node
    capacity: current capacity of the vehicle
    invalid_mask: binary mask (0/1 <--> legal/illegal)
    """

    problem: Array  # (problem_size + 1, 3)
    position: jnp.int32
    capacity: jnp.float32
    action_mask: Array  # (problem_size + 1,)
