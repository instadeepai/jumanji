# MMST Environment

<p align="center">
        <img src="../env_anim/mmst.gif" width="600"/>
</p>

The cooperative minimum spanning tree (mmst) environment consists of a random connected graph
with groups of nodes (same node types) that needs to be connected.
The goal of the environment is to connect all nodes of the same type together
without using the same utility nodes (nodes that do not belong to any group of nodes).

An episode ends when all group of nodes are connected or the maximum number of steps is reached.

> Note:
>
> This environment can be treated as a multi agent problem with each agent atempting to connect
> one group of node. In this implementation, we treat the problem as single agent that outputs
> multiple actions per nodes.


## Observation
At each step observation contains 4 items: a node_types, an adjacency matrix for the graph,
an action mask for each group of nodes (agent) and current node positon of each agent.

- `node_types`: Array representing the types of nodes in the problem.
        For example, if we have 12 nodes, their indices are 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11.
        Let's consider we have 2 agents. Agent 0 wants to connect nodes (0, 1, 9),
        and agent 1 wants to connect nodes (3, 5, 8).
        The remaining nodes are considered utility nodes.
        Therefore, in the state view, the node_types are
        represented as [0, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, -1].
        When generating the problem, each agent starts from one of its nodes.
        So, if agent 0 starts on node 1 and agent 1 on node 3,
        the connected_nodes array will have values [1, -1, ...] and [3, -1, ...] respectively.
        The agent's observation is represented using the following rules:
        - Each agent should see its connected nodes on the path as 0.
        - Nodes that the agent still needs to connect are represented as 1.
        - The next agent's nodes are represented by 2 and 3, the next by 4 and 5, and so on.
        - Utility unconnected nodes are represented by -1.
        For the 12 node example mentioned above,
        the expected observation view node_types will have the following values:
        node_types = jnp.array(
            [
                [1, 0, -1, 2, -1, 3, 1, -1, 3, 1, -1, -1],
                [3, 2, -1, 0, -1, 1, 3, -1, 1, 3, -1, -1],
            ],
            dtype=jnp.int32,
        )
        Note: to make the environment single agent, we use the first agent's observation.

 - `adj_matrix`: Adjacency matrix representing the connections between nodes.

 - `positions`: Current node positions of the agents.
        In our current problem, this will be represented as jnp.array([1, 3]).

-  `action_masks`: Binary mask indicating the validity of each action.
        Given the current node on which the agent is located,
        this mask determines if there is a valid edge to every other node.


## Action
The action space is a `MultiDiscreteArray` of shape `(num_agents,)` of integer values in the range
of `[0, num_nodes-1]`. During every step, an agent picks the next node it wants to move to.
An action is invalid if the agent picks a node it has no edge to or the node is a utility node already
been used by another agent.


## Reward
An agent recieves a reward of 0.1 every step it gets a valid connection, a reward of -0.03 if it does not
connect and an extra penalty of -0.01 if choses an invalid action.

The total step reward is the sum of rewards per agent.


## Registered Versions 📖
- `MMST-v0`, 2 agents, 12 nodes, 24 edges and 3 nodes to connect per agent.
