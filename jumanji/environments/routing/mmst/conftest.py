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

from typing import Tuple

import jax
import jax.numpy as jnp
import pytest

from jumanji.environments.routing.mmst.env import MMST
from jumanji.environments.routing.mmst.generator import SplitRandomGenerator
from jumanji.environments.routing.mmst.types import State
from jumanji.environments.routing.mmst.utils import build_adjecency_matrix
from jumanji.types import TimeStep, restart


@pytest.fixture(scope="module")
def mmst_split_gn_env() -> MMST:
    """Instantiates a default `MMST` environment."""
    return MMST(
        generator_fn=None,
        reward_fn=None,
    )


@pytest.fixture(scope="module")
def deterministic_mmst_env() -> Tuple[MMST, State, TimeStep]:
    """Instantiates a `MMST` environment."""

    num_nodes_per_agent = 3

    env = MMST(
        generator_fn=SplitRandomGenerator(
            num_nodes=12,
            num_edges=18,
            max_degree=5,
            num_agents=2,
            num_nodes_per_agent=num_nodes_per_agent,
            max_step=12,
        ),
        reward_fn=None,
        step_limit=12,
    )

    state, timestep = env.reset(jax.random.PRNGKey(10))

    key = jax.random.PRNGKey(0)

    num_agents = 2
    nodes_to_connect = jnp.array([[0, 1, 6], [3, 5, 8]], dtype=jnp.int32)

    edges = jnp.array(
        [
            [0, 1],
            [0, 3],
            [1, 2],
            [1, 4],
            [2, 4],
            [2, 5],
            [3, 4],
            [3, 7],
            [4, 5],
            [4, 8],
            [6, 7],
            [6, 10],
            [7, 8],
            [8, 9],
            [8, 10],
            [9, 10],
            [9, 11],
        ],
        dtype=jnp.int32,
    )

    adj_matrix = build_adjecency_matrix(12, edges)

    node_edges = jnp.ones((12, 12)) * -1
    node_edges = node_edges.at[0, [1, 3]].set(jnp.array([1, 3]))
    node_edges = node_edges.at[1, [0, 2, 4]].set(jnp.array([0, 2, 4]))
    node_edges = node_edges.at[2, [1, 4, 5]].set(jnp.array([1, 4, 5]))
    node_edges = node_edges.at[3, [0, 4, 7]].set(jnp.array([0, 4, 7]))
    node_edges = node_edges.at[4, [1, 2, 3, 5, 8]].set(jnp.array([1, 2, 3, 5, 8]))
    node_edges = node_edges.at[5, [2, 4]].set(jnp.array([2, 4]))
    node_edges = node_edges.at[6, [7, 10]].set(jnp.array([7, 10]))
    node_edges = node_edges.at[7, [3, 6, 8]].set(jnp.array([3, 6, 8]))
    node_edges = node_edges.at[8, [4, 7, 9, 10]].set(jnp.array([4, 7, 9, 10]))
    node_edges = node_edges.at[9, [8, 10, 11]].set(jnp.array([8, 10, 11]))
    node_edges = node_edges.at[10, [6, 8, 9]].set(jnp.array([6, 8, 9]))
    node_edges = node_edges.at[11, [9]].set(jnp.array([9]))

    node_edges = jnp.array(node_edges, dtype=jnp.int32)

    node_types = jnp.array([0, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, -1], dtype=jnp.int32)

    conn_nodes = -1 * jnp.ones((2, 12), dtype=jnp.int32)
    conn_nodes = conn_nodes.at[0, 0].set(1)
    conn_nodes = conn_nodes.at[1, 0].set(3)

    conn_nodes_index = -1 * jnp.ones((2, 12), dtype=jnp.int32)
    conn_nodes_index = conn_nodes_index.at[0, 1].set(1)
    conn_nodes_index = conn_nodes_index.at[1, 3].set(3)

    positions = jnp.array([1, 3], dtype=jnp.int32)

    active_node_edges = jnp.repeat(node_edges[None, ...], num_agents, axis=0)
    active_node_edges = env._update_active_edges(
        active_node_edges, positions, node_types
    )
    finished_agents = jnp.zeros((num_agents), dtype=bool)

    state = State(
        node_types=node_types,
        adj_matrix=adj_matrix,
        connected_nodes=conn_nodes,
        connected_nodes_index=conn_nodes_index,
        nodes_to_connect=nodes_to_connect,
        position_index=jnp.zeros((num_agents), dtype=jnp.int32),
        positions=positions,
        node_edges=active_node_edges,
        action_mask=env._make_action_mask(
            active_node_edges, positions, finished_agents
        ),
        finished_agents=finished_agents,
        step_count=jnp.array(0, int),
        key=key,
    )

    timestep = restart(observation=env._state_to_observation(state), shape=num_agents)

    return env, state, timestep
