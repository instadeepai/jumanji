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

from abc import ABC, abstractmethod
from typing import Tuple

import chex
import jax
from jax import numpy as jnp

from jumanji.environments.routing.mmst.constants import EMPTY_NODE
from jumanji.environments.routing.mmst.types import State
from jumanji.environments.routing.mmst.utils import (
    build_adjecency_matrix,
    make_action_mask,
    multi_random_walk,
    update_active_edges,
)


class Generator(ABC):
    """Base class for generators for the `MMST` environment."""

    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        max_degree: int,
        num_agents: int,
        num_nodes_per_agent: int,
        max_step: int,
    ) -> None:
        """Initialises a graph generator.

        Args:
            num_nodes: number of nodes in the graph.
            num_edges: number of edges in the graph.
            max_degree: maximum degree a node can have.
            num_agents: number of agents.
            num_nodes_per_agent: number of nodes to connect per agent.
        """
        self._num_nodes = num_nodes
        self._num_edges = num_edges
        self._max_degree = max_degree
        self._num_agents = num_agents
        self._num_nodes_per_agent = num_nodes_per_agent
        self._total_comps = num_nodes_per_agent * num_agents
        self._max_step = max_step

        if num_nodes_per_agent * num_agents > num_nodes * 0.8:
            raise ValueError(
                f"The number of nodes to connect i.e. {num_nodes_per_agent * num_agents} "
                f"should be much less than than 80% of the total number of nodes. "
                f"This is to guarantee there are enough remaining nodes to "
                f"create a path with all the nodes we want to connect."
            )

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def num_nodes_per_agent(self) -> int:
        return self._num_nodes_per_agent

    @abstractmethod
    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a random graph and different nodes to connect per agents.

        Returns:
            a `MMST` environment state
        """


class SplitRandomGenerator(Generator):
    """Generates a random environments that is solvable by spliting the graph into multiple sub graphs.

    Returns a graph and with a desired number of edges and nodes to connect per agent.
    """

    def __init__(
        self,
        num_nodes: jnp.int32,
        num_edges: jnp.int32,
        max_degree: jnp.int32,
        num_agents: jnp.int32,
        num_nodes_per_agent: jnp.int32,
        max_step: jnp.int32,
    ) -> None:
        super().__init__(
            num_nodes, num_edges, max_degree, num_agents, num_nodes_per_agent, max_step
        )

    def __call__(self, key: chex.PRNGKey) -> State:
        graph_key, key = jax.random.split(key)

        # Generate a random graph.
        adj_matrix, node_edges, nodes_per_sub_graph = self._generate_graph(graph_key)

        # Initialise empty arrays for the different states.
        (
            state_nodes_to_connect,
            node_types,
            conn_nodes,
            conn_nodes_index,
            agents_pos,
        ) = self._initialise_states()

        # Populate the states.
        for agent in range(self._num_agents):
            select_key, key = jax.random.split(key)
            agent_components = jax.random.choice(
                select_key,
                nodes_per_sub_graph[agent],
                [self._num_nodes_per_agent],
                replace=False,
            )
            node_types = node_types.at[agent_components].set(agent)
            agents_pos = agents_pos.at[agent].set(agent_components[0])
            conn_nodes = conn_nodes.at[agent, 0].set(agent_components[0])
            conn_nodes_index = conn_nodes_index.at[agent, agent_components[0]].set(
                agent_components[0]
            )
            state_nodes_to_connect = state_nodes_to_connect.at[agent].set(
                agent_components
            )

        active_node_edges = jnp.repeat(node_edges[None, ...], self.num_agents, axis=0)
        active_node_edges = update_active_edges(
            self.num_agents, active_node_edges, agents_pos, node_types
        )
        finished_agents = jnp.zeros((self.num_agents), dtype=bool)

        state = State(
            node_types=node_types,
            adj_matrix=adj_matrix,
            nodes_to_connect=state_nodes_to_connect,
            connected_nodes=conn_nodes,
            connected_nodes_index=conn_nodes_index,
            position_index=jnp.zeros((self.num_agents), dtype=jnp.int32),
            positions=agents_pos,
            node_edges=active_node_edges,
            action_mask=make_action_mask(
                self.num_agents,
                self.num_nodes,
                active_node_edges,
                agents_pos,
                finished_agents,
            ),
            finished_agents=finished_agents,
            step_count=jnp.array(0, int),
            key=key,
        )

        return state

    def _generate_graph(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:

        nodes = jnp.arange(self._num_nodes, dtype=jnp.int32)
        graph, nodes_per_sub_graph = multi_random_walk(
            nodes, self._num_edges, self._num_agents, self._max_degree, key
        )

        node_edges = graph.node_edges
        adj_matrix = build_adjecency_matrix(self._num_nodes, graph.edges)

        return adj_matrix, node_edges, nodes_per_sub_graph

    def _initialise_states(
        self,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """Initialises arrays to hold environment states"""

        state_nodes_to_connect = EMPTY_NODE * (
            jnp.ones((self._num_agents, self._num_nodes_per_agent), dtype=jnp.int32)
        )

        node_types = EMPTY_NODE * jnp.ones(self._num_nodes, dtype=jnp.int32)
        conn_nodes = EMPTY_NODE * jnp.ones(
            (self._num_agents, self._max_step), dtype=jnp.int32
        )
        conn_nodes_index = EMPTY_NODE * jnp.ones(
            (self._num_agents, self._num_nodes), dtype=jnp.int32
        )

        agents_pos = jnp.zeros((self._num_agents), dtype=jnp.int32)

        return (
            state_nodes_to_connect,
            node_types,
            conn_nodes,
            conn_nodes_index,
            agents_pos,
        )
