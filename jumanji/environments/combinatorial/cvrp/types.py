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

import jax.random

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex
import jax.numpy as jnp
from chex import Array


@dataclass
class State:
    """
    coordinates: array with the coordinates of all nodes (+ depot)
    demands: array with the demands of all nodes (+ depot)
    position: index of the current node
    capacity: current capacity of the vehicle
    visited_mask: binary mask (False/True <--> unvisited/visited)
    order: array of node indices denoting route (-1 --> not filled yet)
    num_total_visits: number of performed visits (it can count depot multiple times)
    """

    coordinates: Array  # (problem_size + 1, 2)
    demands: Array  # (problem_size + 1,)
    position: jnp.int32
    capacity: jnp.int32
    visited_mask: Array  # (problem_size + 1,)
    order: Array  # (2 * problem_size,) - this size is worst-case (visit depot after each node)
    num_total_visits: jnp.int32
    key: chex.PRNGKey = jax.random.PRNGKey(0)


class Observation(NamedTuple):
    """
    coordinates: array with the coordinates of all nodes (+ depot)
    demands: array with the demands of all nodes (+ depot)
    position: index of the current node
    capacity: current capacity of the vehicle
    action_mask: binary mask (True/False <--> invalid/valid action)
    """

    coordinates: Array  # (problem_size + 1, 2)
    demands: Array  # (problem_size + 1,)
    position: jnp.int32
    capacity: jnp.float32
    action_mask: Array  # (problem_size + 1,)
