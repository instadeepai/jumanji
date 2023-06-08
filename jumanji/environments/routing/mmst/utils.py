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

from typing import Any, List, Tuple

import chex
import jax
import numpy as np
from jax import numpy as jnp

from jumanji.environments.routing.mmst.constants import (
    EMPTY_EDGE,
    EMPTY_NODE,
    UTILITY_NODE,
)
from jumanji.environments.routing.mmst.types import Graph


def build_adjecency_matrix(num_nodes: int, edges: jnp.ndarray) -> jnp.ndarray:
    """Build adjaceny matrix from an array with edges."""

    adj_matrix = jnp.zeros((num_nodes, num_nodes), dtype=int)
    adj_matrix = adj_matrix.at[edges[:, 0], edges[:, 1]].set(1)
    adj_matrix = adj_matrix.at[edges[:, 1], edges[:, 0]].set(1)

    return adj_matrix


def update_active_edges(
    num_agents: int,
    node_edges: chex.Array,
    position: chex.Array,
    node_types: chex.Array,
) -> chex.Array:
    """Update the active agent nodes available to each agent

    Args:
        num_agents: (int) number of agents
        node_edges: (array) with node edges
        position: (array) for current agent position
        node_types: array
    Returns:
        active_node_edges: (array)
    """

    def update_edges(node_edges: chex.Array, node: jnp.int32) -> chex.Array:
        zero_mask = node_edges != node
        ones_inds = node_edges == node
        upd_edges = node_edges * zero_mask - ones_inds
        return upd_edges

    active_node_edges = jnp.copy(node_edges)

    for agent in range(num_agents):
        node = position[agent]
        cond = node_types[node] == UTILITY_NODE

        for agent2 in range(num_agents):
            if agent != agent2:
                upd_edges = jax.lax.cond(
                    cond,
                    update_edges,
                    lambda _edgs, _node: _edgs,
                    active_node_edges[agent2],
                    node,
                )
                active_node_edges = active_node_edges.at[agent2].set(upd_edges)

    return active_node_edges


def make_action_mask(
    num_agents: int,
    num_nodes: int,
    node_edges: chex.Array,
    position: chex.Array,
    finished_agents: chex.Array,
) -> chex.Array:
    """Intialise and action mask for every node based on all its edges

    Args:
        num_agents (int): number of agents
        num_nodes (int): number of nodes
        node_edges (Array): Array with the respective edges for
            every node (-1 for invalid edge)
        position: current node of each agent
        finished_agents: (Array): used to mask finished agents
    Returns:
        action_mask (Array): action mask for each agent at it current node position
    """

    full_action_mask = node_edges != EMPTY_NODE
    action_mask = (
        full_action_mask[jnp.arange(num_agents), position]
        & ~finished_agents[:, jnp.newaxis]
    )

    return action_mask


def check_num_edges(nodes: List[int], num_edges: jnp.int32) -> None:
    """Checks that the number of requested edges is acceptable."""

    num_nodes = len(nodes)

    # Check mininum number of edges is satisfied.
    min_edges = num_nodes - 1
    if num_edges < min_edges:
        raise ValueError("num_edges less than minimum (%i)" % min_edges)

    # Check maximum number of edges is satisfied.
    max_edges = num_nodes * (num_nodes - 1) / 2
    if num_edges > max_edges:
        raise ValueError("num_edges greater than maximum (%i)" % max_edges)


def get_edge_code(edge: chex.Array) -> jnp.float32:
    """Computes a unique code for every edge in the graph.
    We use this function to check if a given edge is already
    added to the graph.
    http://en.wikipedia.org/wiki/Pairing_function
    """

    node_a, node_b = edge[0], edge[1]
    edge_code = 0.5 * (node_a + node_b) * (node_a + node_b + 1) + node_b
    return edge_code


def get_edge_nodes_from_code(z: jnp.float32) -> Tuple[int, int]:
    """Inverse of Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Inverting_the_Cantor_pairing_function
    """

    w = jnp.floor((jnp.sqrt(8 * z + 1) - 1) / 2)
    t = (w**2 + w) / 2
    y = jnp.array(z - t, dtype=int)
    x = jnp.array(w - y, dtype=int)

    return x, y


def correct_edge_code_offset(code: jnp.float32, offset: jnp.int32) -> jnp.float32:
    """Compute the correct edge codes by inverting the correct code
    and adding the offset to the nodes."""

    x, y = get_edge_nodes_from_code(code)
    edge = jnp.array([x, y], dtype=jnp.int32) + offset
    correct_code = get_edge_code(edge)

    return correct_code


def init_graph(nodes: chex.Array, max_degree: int, num_edges: int) -> Graph:
    """Initialise an empty graph.

    Args:
        nodes: array with nodes labels (indices from 0 to num_nodes-1).
        max_degree: highest degree a node can have.
        num_edges: total number of desired edges in the graph.

    Returns:
        graph: an empty Graph.
    """

    num_nodes = len(nodes)

    edges = jnp.ones((num_edges, 2), dtype=jnp.int32) * EMPTY_EDGE
    edge_codes = jnp.ones((num_edges), dtype=jnp.float32) * EMPTY_EDGE
    node_degree = jnp.zeros(num_nodes, dtype=jnp.int32)
    node_edges = jnp.ones((num_nodes, num_nodes), dtype=jnp.int32) * EMPTY_NODE
    edge_index = 0

    graph = Graph(
        nodes=nodes,
        edges=edges,
        edge_codes=edge_codes,
        max_degree=max_degree,
        node_degree=node_degree,
        edge_index=edge_index,
        node_edges=node_edges,
    )

    return graph


def init_graph_merge(
    graph_a: Graph, graph_b: Graph, num_edges: int, max_degree: int
) -> Graph:
    """Merge two graphs and initialize the setting to add new edges.

    Args:
        graph_a: The first graph to merge.
        graph_b: The second graph to merge.
        num_edges: The number of edges in the merged graph.
        max_degree: The maximum degree of any node in the merged graph.

    Returns:
        A new graph representing the merged graph of graph_a and graph_b.
    """

    edges = jnp.ones((num_edges, 2), dtype=jnp.int32) * EMPTY_EDGE
    num_edges_a = graph_a.edges.shape[0]
    num_edges_b = graph_b.edges.shape[0]
    num_edges_ab = num_edges_a + num_edges_b

    edges = edges.at[0:num_edges_a].set(graph_a.edges)
    edges = edges.at[num_edges_a:num_edges_ab].set(graph_b.edges)
    edge_index = num_edges_a + num_edges_b

    edge_codes = jnp.ones((num_edges), dtype=jnp.float32) * EMPTY_EDGE
    edge_codes = edge_codes.at[0:num_edges_a].set(graph_a.edge_codes)
    edge_codes = edge_codes.at[num_edges_a:num_edges_ab].set(graph_b.edge_codes)

    # Nodes handling.
    nodes = jnp.append(graph_a.nodes, graph_b.nodes)
    total_nodes = len(nodes)
    nodes_a = len(graph_a.nodes)
    node_degree = jnp.append(graph_a.node_degree, graph_b.node_degree, axis=0)

    node_edges = jnp.ones((total_nodes, total_nodes), dtype=jnp.int32) * EMPTY_NODE
    node_edges = node_edges.at[0:nodes_a, 0:nodes_a].set(graph_a.node_edges)
    node_edges = node_edges.at[nodes_a:total_nodes, nodes_a:total_nodes].set(
        graph_b.node_edges
    )

    graph = Graph(
        nodes=nodes,
        edges=edges,
        edge_codes=edge_codes,
        max_degree=max_degree,
        node_degree=node_degree,
        edge_index=edge_index,
        node_edges=node_edges,
    )

    return graph


def correct_graph_offset(graph: Graph, offset: int) -> Graph:
    """Correct the node offset applied when generating split graphs.

    To generate a solvable problem, we split the graphs into subgraphs.
    For example, if we want a fully connected graph with 12 nodes and 2 agents,
    we first construct two subgraphs with 6 nodes each.
    Each graph will have nodes labeled [0, 1, 2, ..., 5].
    Before merging the two graphs, we need to rename the nodes of one graph to be [6, 7, ..., 11].
    To accomplish this, we use the offset parameter,
    which in this case will be 6, to perform the renaming.
    """

    nodes = graph.nodes + offset
    edges = graph.edges + offset

    edge_codes = jax.vmap(correct_edge_code_offset, in_axes=(0, None))(
        graph.edge_codes, offset
    )

    node_edges = graph.node_edges
    zero_mask = node_edges != EMPTY_NODE
    ones_mask = node_edges == EMPTY_NODE
    node_edges += offset
    node_edges *= zero_mask
    node_edges += ones_mask * EMPTY_NODE

    graph = Graph(
        nodes=nodes,
        edges=edges,
        edge_codes=edge_codes,
        max_degree=graph.max_degree,
        node_degree=graph.node_degree,
        edge_index=graph.edge_index,
        node_edges=node_edges,
    )

    return graph


def add_edge(graph: Graph, edge: chex.Array) -> Tuple[Graph, bool]:
    """Add the provided edge to the graph."""

    def _add_edge(
        edge: chex.Array, edge_arr: chex.Array, edge_code: jnp.float32, graph: Graph
    ) -> Tuple[Graph, bool]:
        edges = graph.edges.at[graph.edge_index, :].set(edge_arr)
        edge_codes = graph.edge_codes.at[graph.edge_index].set(edge_code.squeeze())
        edge_index = graph.edge_index + 1

        node_edges = graph.node_edges
        node_edges = node_edges.at[edge[0], edge[1]].set(edge[1])
        node_edges = node_edges.at[edge[1], edge[0]].set(edge[0])

        node_degree = graph.node_degree
        node_degree = node_degree.at[edge[0]].set(graph.node_degree[edge[0]] + 1)
        node_degree = node_degree.at[edge[1]].set(graph.node_degree[edge[1]] + 1)

        graph = Graph(
            nodes=graph.nodes,
            edges=edges,
            edge_codes=edge_codes,
            max_degree=graph.max_degree,
            edge_index=edge_index,
            node_edges=node_edges,
            node_degree=node_degree,
        )
        return graph, True

    edge_code = get_edge_code(edge)
    edge_arr = jnp.array([edge[0], edge[1]], dtype=jnp.int32).reshape(
        2,
    )

    # Below we use the edge_code to first check if the edge we want to add
    # has not yet been added.
    # Next we check if the degree of both nodes will not exceed the maximum degree.
    # If these two conditions are not satisfied the edge is not added.

    codes = jnp.sum(graph.edge_codes == edge_code)
    left_deg = jnp.sum(graph.node_degree[edge[0]] > graph.max_degree)
    right_deg = jnp.sum(graph.node_degree[edge[1]] > graph.max_degree)

    cond = jnp.sum(codes == 0) & jnp.sum(left_deg == 0) & jnp.sum(right_deg == 0)

    graph, success = jax.lax.cond(
        cond, _add_edge, lambda *_: (graph, False), edge, edge_arr, edge_code, graph
    )

    return graph, success


def make_random_edge(graph: Graph, key: chex.PRNGKey) -> Tuple[int, int]:
    """Generate a random edge between any two nodes in the graph."""

    random_edge = jax.random.choice(key, graph.nodes, [2], replace=False)
    return (random_edge[0], random_edge[1])


def make_random_edge_from_nodes(
    nodes_a: chex.Array, nodes_b: chex.Array, key: chex.PRNGKey
) -> Tuple[int, int]:
    """Generate a random edge from two sets of nodes."""

    node_a_key, node_b_key = jax.random.split(key)
    a = jax.random.choice(node_a_key, nodes_a)
    b = jax.random.choice(node_b_key, nodes_b)
    edge = (a, b)

    return edge


def add_random_edges(
    graph: Graph, total_edges: jnp.int32, base_key: chex.PRNGKey
) -> Graph:
    """Add random edges until the number of desired edges is reached."""

    def desired_num_edges_not_reach(arg: Any) -> Any:
        graph, total_edges, base_key = arg
        return graph.edge_index < total_edges

    def add_new_edge(arg: Any) -> Any:
        graph, total_edges, base_key = arg
        current_key, base_key = jax.random.split(base_key)
        graph, success = add_edge(graph, make_random_edge(graph, current_key))

        return (graph, total_edges, base_key)

    graph, _, _ = jax.lax.while_loop(
        desired_num_edges_not_reach, add_new_edge, (graph, total_edges, base_key)
    )

    return graph


def update_conected_nodes(
    graph: Graph, edge: chex.Array, source: chex.Array, target: chex.Array
) -> Any:
    """Build a connected graph by creating random edges with all the nodes in the graph."""

    order_edge = jax.lax.cond(
        jnp.all(edge[0] < edge[1]),
        lambda *_: (edge[0], edge[1]),
        lambda *_: (edge[1], edge[0]),
        edge,
    )

    graph, success = add_edge(graph, order_edge)

    source, target = jax.lax.cond(
        success,
        lambda *_: (source.at[edge[1]].set(-1), target.at[edge[1]].set(edge[1])),
        lambda *_: (source, target),
        source,
        target,
    )

    return (source, target, graph)


def dummy_add_nodes(
    graph: Graph, edge: chex.Array, source: chex.Array, target: chex.Array
) -> Any:
    return (source, target, graph)


def random_walk(
    nodes: chex.Array, num_edges: jnp.int32, max_degree: jnp.int32, key: chex.PRNGKey
) -> Graph:
    """Create a uniform spanning tree (UST) using a random walk,
    then add random edges until the desired number of edges is reached.
    https://en.wikipedia.org/wiki/Uniform_spanning_tree

    Args:
        nodes: Array with node labels (indices from 0 to num_nodes-1).
        num_edges: Total number of desired edges in the graph.
        max_degree: The highest degree a node can have.
        key: Base key from which all other keys are generated for any random sampling.

    Returns:
        graph: A random fully connected graph with
             the desired number of nodes, number of edges, and maximum degree.
    """

    check_num_edges(nodes, num_edges)

    # Create two partitions, source and target. Initially store all nodes in source.
    source = jnp.copy(nodes)
    target = jnp.ones_like(source) * -1

    # Pick a random node, mark it as visited and make it the current node.
    current_key, base_key = jax.random.split(key)
    current_node = jax.random.randint(
        current_key, (1,), minval=0, maxval=len(nodes) - 1, dtype=jnp.int32
    )
    source = source.at[current_node].set(-1)
    target = target.at[current_node].set(current_node)

    graph = init_graph(nodes, max_degree, num_edges)

    def all_nodes_not_added(arg: Any) -> Any:
        """Check if all nodes have already been added."""

        source, *_ = arg
        return jnp.sum(source != -1) > 0

    def add_an_edge_with_a_new_node(arg: Any) -> Any:
        source, target, graph, current_node, base_key = arg
        # Randomly pick the next node from the neighbors of the current node.
        # As we are generating a connected graph, we assume a complete graph.
        current_key, base_key = jax.random.split(base_key)
        neighbor_node = jax.random.choice(current_key, nodes, [1])

        edge = (current_node, neighbor_node)
        # If the new node hasn't been visited, add the edge from current to new.
        is_valid = jnp.sum(jnp.all(target[neighbor_node] == -1))

        source, target, graph = jax.lax.cond(
            is_valid,
            update_conected_nodes,
            dummy_add_nodes,
            graph,
            edge,
            source,
            target,
        )

        current_node = neighbor_node

        return (source, target, graph, current_node, base_key)

    # We first create a connected graph by adding edges with all the nodes.
    source, target, graph, current_node, base_key = jax.lax.while_loop(
        all_nodes_not_added,
        add_an_edge_with_a_new_node,
        (source, target, graph, current_node, base_key),
    )

    # Add random edges until the number of desired edges is reached.
    graph = add_random_edges(graph, num_edges, base_key)

    return graph


def merge_graphs(
    graph_a: Graph,
    graph_b: Graph,
    num_edges: jnp.int32,
    max_degree: jnp.int32,
    base_key: chex.PRNGKey,
) -> Graph:
    """Merge two graphs, then randomly add edges by selecting pair of nodes
    with one node from each graph until the desired number of edges is
    reached.

    Args:
        graph_a: first graph.
        graph_b: second graph.
        num_edges: desired number of edges.
        base_key: base key for random selections.

    Returns:
        graph: the merge graph with the desired number of edges.
    """

    graph = init_graph_merge(graph_a, graph_b, num_edges, max_degree)

    base_key1, base_key2 = jax.random.split(base_key, 2)

    # Add one edge between both subgraphs to guarentee the new graph is connected.
    graph, _ = add_edge(
        graph, make_random_edge_from_nodes(graph_a.nodes, graph_b.nodes, base_key1)
    )

    # Add remaining edges until the desired number of edges is reached.
    graph = add_random_edges(graph, num_edges, base_key2)

    return graph


def multi_random_walk(
    nodes: chex.Array,
    num_edges: jnp.int32,
    num_agents: jnp.int32,
    max_degree: jnp.int32,
    key: chex.PRNGKey,
) -> Tuple[Graph, List[chex.Array]]:
    """Create a uniform spanning tree (UST) using a random walk by combining multiple spanning trees
    using random edges among the nodes of the different spanning trees.

    Args:
        nodes: Array with node labels (indices from 0 to num_nodes-1).
        num_edges: Total number of desired edges in the graph.
        max_degree: The highest degree a node can have.
        num_agents: Number of subtrees to generate before merging.
        key: Base key from which all other keys are generated for any random sampling.

    Returns:
        graph: A random fully connected graph with
           the desired number of nodes, number of edges, and maximum degree.
    """

    check_num_edges(nodes, num_edges)

    nodes_per_sub_graph = jnp.array_split(nodes, num_agents)
    nodes_offsets = [nodes_i[0] for nodes_i in nodes_per_sub_graph]
    nodes_per_sub_graph_offset = [
        nodes_per_sub_graph[i] - nodes_offsets[i] for i in range(num_agents)
    ]
    num_nodes_per_sub_graph = [len(nodes_i) for nodes_i in nodes_per_sub_graph]

    # We use the following code to compute the number of edges per subgraph.
    # min_edges = num_nodes - 1
    # max_edges = num_nodes * (num_nodes - 1) // 2
    # mean_edges = num_edges // num_agents // 2
    # num_edges_per_sub_graph = min(max(min_edges, mean_edges), max_edges

    total_edges_sub_graph = num_edges // 2
    edges_per_sub_graph = jnp.array_split(jnp.arange(total_edges_sub_graph), num_agents)
    num_edges_per_sub_graph_limits = [
        [num_nodes_i - 1, (num_nodes_i * (num_nodes_i - 1)) // 2]
        for num_nodes_i in num_nodes_per_sub_graph
    ]

    num_edges_per_sub_graph = [0] * num_agents
    for index in range(num_agents):
        num_edges_per_sub_graph[index] = min(
            num_edges_per_sub_graph_limits[index][1],
            max(
                num_edges_per_sub_graph_limits[index][0],
                len(edges_per_sub_graph[index]),
            ),
        )

    total_edges_merge_graph = num_edges - sum(num_edges_per_sub_graph)

    graph_key, base_key = jax.random.split(key)
    sub_graph_keys = jax.random.split(graph_key, num_agents)

    graphs = [
        random_walk(
            nodes_per_sub_graph_offset[i],
            num_edges_per_sub_graph[i],
            max_degree,
            sub_graph_keys[i],
        )
        for i in range(num_agents)
    ]

    # Get the total number of edges we need to add when merging the graphs.
    sum_ratio: int = np.sum(np.arange(1, num_agents))
    frac = np.cumsum(
        [total_edges_merge_graph * (i) / sum_ratio for i in range(1, num_agents - 1)]
    )
    edges_per_merge_graph = jnp.split(jnp.arange(total_edges_merge_graph), frac)
    num_edges_per_merge_graph = [len(edges) for edges in edges_per_merge_graph]

    # Merge the graphs.
    graph, *graphs = graphs
    total_edges = num_edges_per_sub_graph[0]
    merge_graph_keys = jax.random.split(base_key, num_agents - 1)

    for i in range(1, num_agents):
        total_edges += num_edges_per_sub_graph[i] + num_edges_per_merge_graph[i - 1]
        graph_i = correct_graph_offset(graphs[i - 1], nodes_offsets[i])
        graph = merge_graphs(
            graph, graph_i, total_edges, max_degree, merge_graph_keys[i]
        )

    return graph, nodes_per_sub_graph
