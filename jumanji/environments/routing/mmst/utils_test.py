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

from functools import partial

import jax
import jax.numpy as jnp
import pytest

from jumanji.environments.routing.mmst.utils import (
    build_adjecency_matrix,
    check_num_edges,
    correct_edge_code_offset,
    get_edge_code,
    get_edge_nodes_from_code,
    multi_random_walk,
    random_walk,
)


def test__adj_matrix_construction() -> None:
    num_nodes = 5
    edges = jnp.array([[0, 1], [1, 2], [2, 3], [3, 4]])

    adj_matrix = build_adjecency_matrix(num_nodes, edges)

    expected_adj_matrix = jnp.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
        ],
        dtype=int,
    )

    assert jnp.array_equal(adj_matrix, expected_adj_matrix)


def test__check_num_edges() -> None:
    """Test if the check for the minimum and maximum
    number of edges works as expected."""

    num_nodes = 5
    nodes = jnp.arange(num_nodes)
    min_edges = num_nodes - 1
    check_num_edges(nodes, min_edges)
    max_edges = num_nodes * (num_nodes - 1) / 2
    check_num_edges(nodes, max_edges)

    with pytest.raises(ValueError):
        check_num_edges(nodes, min_edges - 1)
        check_num_edges(nodes, max_edges + 1)


def test__edge_code() -> None:
    """Test if the paring function and its inverse are correctly implemented."""

    edge = jnp.array([20, 34])
    code = get_edge_code(edge)
    x, y = get_edge_nodes_from_code(code)

    edge_out = jnp.array([x, y])

    assert jnp.array_equal(edge, edge_out)

    code_off = get_edge_code(edge - 10)
    corret_code = correct_edge_code_offset(code_off, 10)
    assert code == corret_code


def test__graph_walk() -> None:
    """Test the main graph generation functions."""

    num_nodes = 12
    num_edges = 66
    max_degree = 11
    num_agents = 2
    nodes = jnp.arange(num_nodes, dtype=jnp.int32)

    key = jax.random.PRNGKey(0)
    graph_keys = jax.random.split(key, 10)

    random_walk_jit = jax.jit(partial(random_walk, nodes, num_edges, max_degree))
    multi_random_walk_jit = jax.jit(
        partial(multi_random_walk, nodes, num_edges, num_agents, max_degree)
    )

    for graph_key in graph_keys:
        _ = random_walk_jit(graph_key)

    for graph_key in graph_keys:
        _ = multi_random_walk_jit(graph_key)
