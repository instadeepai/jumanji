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

import chex
import jax.numpy as jnp

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.cmst.types import Observation, State
from jumanji.types import Action, TimeStep


class CoopMinSpanTree(Environment[State]):
    """The cooperative minimum spanning tree (CMST) environment consists of a random connected graph
    with groups of nodes (same node types) that need to be connected.
    The goal of the environment is to connect all nodes of the same type together
    without using the utility nodes (nodes that do not belong to any group of nodes).

    Note: routing problems are randomly generated and may not be solvable!
    Additionally, the total number of nodes should be at least 20% more than
    the number of nodes we want to connect. This is to guarantee we have enough remaining
    nodes to create a path with all the nodes we want to connect. In the current implementation,
    the total number of nodes to connect (by all agents) should be less than
    80% of the total number of nodes. An exception will be raised if the number of nodes is
    not greater than (0.8 x num_agents x num_nodes_per_agent).

    - observation: Observation
        - node_type: jax array (int) of shape (num_agents, num_nodes).
            the component type of each node (-1 represents utility nodes).
        - edges: jax array (int) of shape (num_edges, 2).
            all the edges in the graph.
        - position: jax array (int) of shape (num_agents,).
            the index of the last visited node.
        - action_mask: jax array (bool) of shape (num_agent, num_nodes).
            binary mask (False/True <--> invalid/valid action).

    - reward: jax array (float) of shape (num_agents,).
        - each agent's reward.

    - action: jax array (int) of shape (num_agents,): [0,1,..., num_nodes-1]
        Each agent selects the next node to which it wants to connect.

    - state: State
        - node_type: jax array (int) of shape (num_nodes,).
            the component type of each node (-1 represents utility nodes).
        - edges: jax array (int) of shape (num_edges, 2).
            all the edges in the graph.
        - connected_nodes: jax array (int) of shape (num_agents, time_limit).
            we only count each node visit once.
        - connected_nodes_index: jax array (int) of shape (num_agents, num_nodes).
        - position_index: jax array (int) of shape (num_agents,).
        - node_edges: jax array (int) of shape (num_agents, num_nodes, num_nodes).
        - position: jax array (int) of shape (num_agents,).
            the index of the last visited node.
        - action_mask: jax array (bool) of shape (num_agent, num_nodes).
            binary mask (False/True <--> invalid/valid action).
        - finished_agents: jax array (bool) of shape (num_agent,).
        - nodes_to_connect: jax array (int) of shape (num_agents, num_nodes_per_agent).
        - step_count: step counter.
        - time_limit: the number of steps allowed before an episode terminates.
        - key: PRNG key for random sample.

    - constants definitions:
        - Nodes
            - INVALID_NODE = -1: used to check if an agent selects an invalid node.
                A node may be invalid if its has no edge with the current node or if it is a
                utility node already selected by another agent.
            - UTILITY_NODE = -1: utility node (belongs to no agent).
            - EMPTY_NODE = -1: used for padding.
                state.connected_nodes stores the path (all the nodes) visited by an agent. Hence
                it has size equal to the step limit. We use this constant to initialise this array
                since 0 represents the first node.
            - DUMMY_NODE = -10: used for tie-breaking if multiple agents select the same node.

        - Edges
            - EMPTY_EDGE = -1: used for masking edges array.
               state.node_edges is the graph's adjacency matrix, but we don't represent it
               using 0s and 1s, we use the node values instead, i.e `A_ij = j` or `A_ij = -1`.
               Also edges are masked when utility nodes
               are selected by an agent to make it unaccessible by other agents.

        - Actions
            - INVALID_CHOICE = -1
            - INVALID_TIE_BREAK = -2
            - INVALID_ALREADY_TRAVERSED = -3
    """

    def __init__(
        self,
        num_nodes: int = 12,
        num_edges: int = 24,
        max_degree: int = 5,
        num_agents: int = 2,
        num_nodes_per_agent: int = 3,
        time_limit: int = 70,
    ):
        """Create the Cooperative Minimum Spanning Tree environment.

        Args:
            num_nodes: number of nodes in the graph.
            num_edges: number of edges in the graph.
            max_degree: highest degree a node can have.
            num_agents: number of agents.
            num_nodes_per_agent: number of nodes to connect by each agent.
            reward_per_timestep: the reward given to an agent for every timestep
                without any connection.
            reward_for_noop: reward given if an agent performs picks an invalid action
            reward_for_connection: the reward given to an agent for connecting
                any new of its component node.
            time_limit: the number of steps allowed before an episode terminates.

        """

        if num_nodes_per_agent * num_agents > num_nodes * 0.8:
            raise ValueError(
                f"The number of nodes to connect i.e. {num_nodes_per_agent * num_agents} "
                f"should be much less than the number of nodes, which is {int(0.8*num_nodes)}."
            )

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_agents = num_agents
        self.num_nodes_per_agent = num_nodes_per_agent
        self.max_degree = max_degree

    def __repr__(self) -> str:
        return (
            f"CMST(num_nodes={self.num_nodes}, num_edges={self.num_edges}, "
            f"num_agents={self.num_agents}, num_components={self.num_nodes_per_agent})"
            f"max_degree={self.max_degree}"
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec.

        Returns:
            action_spec: a `specs.MultiDiscreteArray` spec.
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.full((self.num_agents,), self.num_nodes, jnp.int32),
            name="action",
        )

    def reward_spec(self) -> specs.Array:
        """Returns the reward spec.

        Returns:
            reward_spec: a `specs.Array` spec.
        """
        return specs.Array(shape=(self.num_agents,), dtype=float, name="reward")

    def discount_spec(self) -> specs.BoundedArray:
        """Returns the discount spec.

        Returns:
            discount_spec: a `specs.Array` spec.
        """
        return specs.BoundedArray(
            shape=(self.num_agents,),
            dtype=float,
            minimum=0.0,
            maximum=1.0,
            name="discount",
        )

    def observation_spec(self) -> specs.Spec:
        """Returns the observation spec.

        Returns:
            observation_spec: a Tuple containing the spec for each of the constituent fields of an
            observation.
        """
        node_types = specs.BoundedArray(
            shape=(self.num_agents, self.num_nodes),
            minimum=-1,
            maximum=self.num_agents * 2 - 1,
            dtype=jnp.int32,
            name="node_types",
        )
        edges = specs.BoundedArray(
            shape=(self.num_agents, self.num_edges, 2),
            minimum=0,
            maximum=self.num_nodes - 1,
            dtype=jnp.int32,
            name="edges",
        )
        position = specs.BoundedArray(
            shape=(self.num_agents,),
            minimum=-1,
            maximum=self.num_nodes - 1,
            dtype=jnp.int32,
            name="position",
        )
        action_mask = specs.BoundedArray(
            shape=(self.num_agents, self.num_nodes),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action mask",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            node_types=node_types,
            edges=edges,
            position=position,
            action_mask=action_mask,
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:  # type: ignore
        pass  # type: ignore

    def step(self, state: State, action: Action) -> Tuple[State, TimeStep]:  # type: ignore
        pass  # type: ignore
