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

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex
from jax import numpy as jnp


class Observation(NamedTuple):
    nodes: chex.Array  # i32[n_nodes, 2], holding the (x,y) coordinates of each node
    edges: chex.Array  # i32[n_edges, 2], holding the two nodes' id of each edge


@dataclass
class State:
    key: chex.PRNGKey
    nodes: chex.Array  # i32[n_nodes, 2]
    edges: chex.Array  # i32[n_nodes, 2]
    step: jnp.int32
