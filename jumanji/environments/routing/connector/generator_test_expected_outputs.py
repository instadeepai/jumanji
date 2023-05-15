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
from jax import numpy as jnp

from jumanji.environments.routing.connector.types import Agent

### grids for testing
empty_grid = jnp.zeros((5, 5))
valid_starting_grid = jnp.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [2, 0, 0, 0, 5],
        [0, 0, 0, 0, 0],
        [0, 0, 8, 0, 0],
    ],
    dtype=int,
)
valid_starting_grid_after_1_step = jnp.array(
    [
        [0, 0, 0, 0, 0],
        [2, 0, 0, 0, 5],
        [1, 0, 0, 0, 4],
        [0, 0, 8, 0, 0],
        [0, 0, 7, 0, 0],
    ],
    dtype=int,
)
valid_end_grid = jnp.array(
    [
        [1, 1, 1, 1, 2],
        [4, 4, 4, 4, 4],
        [7, 7, 7, 0, 5],
        [7, 0, 7, 7, 7],
        [7, 8, 0, 0, 7],
    ],
    dtype=int,
)
valid_end_grid2 = jnp.array(
    [
        [2, 1, 1, 0, 3],
        [5, 0, 1, 1, 1],
        [4, 4, 7, 7, 7],
        [0, 4, 7, 0, 7],
        [6, 4, 9, 0, 8],
    ],
    dtype=int,
)
grid_to_test_available_cells = jnp.array(
    [
        [1, 1, 1, 1, 2],
        [4, 4, 4, 4, 5],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=int,
)
grids_after_1_agent_step = jnp.array(
    [
        [
            [0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [1, 0, 0, 0, 5],
            [0, 0, 0, 0, 0],
            [0, 0, 8, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [2, 0, 0, 0, 4],
            [0, 0, 0, 0, 0],
            [0, 0, 8, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [2, 0, 0, 0, 5],
            [0, 0, 8, 0, 0],
            [0, 0, 7, 0, 0],
        ],
    ],
)
### Agents for testing
agents_finished = Agent(
    id=jnp.array([0, 1, 2]),
    start=jnp.array([[2, 0], [2, 4], [4, 2]]),
    target=jnp.array([[-1, -1], [-1, -1], [-1, -1]]),
    position=jnp.array([[0, 4], [2, 4], [4, 1]]),
)
agents_reshaped_for_generator = Agent(
    id=jnp.arange(3),
    start=jnp.array([[0, 1, 4], [0, 0, 4]]),
    target=jnp.array([[-1, -1, -1], [-1, -1, -1]]),
    position=jnp.array([[0, 4, 4], [4, 0, 2]]),
)
agents_starting = Agent(
    id=jnp.array([0, 1, 2]),
    start=jnp.array([[2, 0], [2, 4], [4, 2]]),
    target=jnp.array([[-1, -1], [-1, -1], [-1, -1]]),
    position=jnp.array([[2, 0], [2, 4], [4, 2]]),
)
agents_starting_move_1_step_up = Agent(
    id=jnp.array([0, 1, 2]),
    start=jnp.array([[2, 0], [2, 4], [4, 2]]),
    target=jnp.array([[-1, -1], [-1, -1], [-1, -1]]),
    position=jnp.array([[1, 0], [1, 4], [3, 2]]),
)
### keys for testing
key = jax.random.PRNGKey(0)
key_1, key_2 = jax.random.split(key)
