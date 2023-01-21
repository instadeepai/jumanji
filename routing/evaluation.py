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

import functools

import jax
import jax.numpy as jnp
from chex import Array

from jumanji.environments.combinatorial.routing.constants import SOURCE
from jumanji.environments.combinatorial.routing.env import Routing


def is_episode_finished(env: Routing, grid: Array) -> jnp.bool_:
    """Returns True if all agents are finished and the episode is completed."""
    dones = env.get_finished_agents(grid)
    return jnp.all(dones)


def proportion_connected(env: Routing, grid: Array) -> float:
    """Calculates the proportion of wires that are connected."""
    connected_agents = jax.vmap(functools.partial(env.is_agent_connected, grid))(
        jnp.arange(env.num_agents)
    )
    proportion: float = jnp.mean(connected_agents).item()
    return proportion


def is_board_complete(env: Routing, grid: Array) -> jnp.bool_:
    """Returns True if all agents in a state are connected.

    Args:
        env: instance of the `Routing` environment.
        grid: Any observation of the environment grid.

    Return:
        True if all agents are connected otherwise False.
    """
    return proportion_connected(env, grid) == 1


def wire_length(env: Routing, grid: Array) -> int:
    """Calculates the length of all the wires on the grid."""
    total_wire_length: int = jnp.sum(
        jax.vmap(lambda i: jnp.count_nonzero(grid == SOURCE + 3 * i))(
            jnp.arange(env.num_agents)
        ),
        dtype=int,
    )
    return total_wire_length
