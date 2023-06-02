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

from jumanji.environments.routing.mmst.constants import UTILITY_NODE
from jumanji.environments.routing.mmst.generator import SplitRandomGenerator
from jumanji.environments.routing.mmst.types import State


def check_generator(params: Tuple, state: State) -> None:
    """Check it the graph data have been generated correctly."""

    (
        num_nodes,
        _,
        _,
        num_agents,
        num_nodes_per_agent,
        max_step,
    ) = params

    assert jnp.min(state.node_types) == UTILITY_NODE
    assert jnp.max(state.node_types) == num_agents - 1
    assert state.positions.shape == (num_agents,)
    assert state.connected_nodes.shape == (num_agents, max_step)
    assert state.connected_nodes_index.shape == (num_agents, num_nodes)
    assert state.node_edges.shape == (num_agents, num_nodes, num_nodes)
    assert state.nodes_to_connect.shape == (num_agents, num_nodes_per_agent)


def test__generator() -> None:
    """Test if the graph generator work as expected."""

    problem_key = jax.random.PRNGKey(0)
    num_nodes = 100
    num_edges = 200
    max_degree = 10
    num_agents = 5
    num_nodes_per_agent = 5
    max_step = 50

    params = (
        num_nodes,
        num_edges,
        max_degree,
        num_agents,
        num_nodes_per_agent,
        max_step,
    )

    generator_fn = SplitRandomGenerator(*params)
    state = generator_fn(problem_key)
    check_generator(params, state)
