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

from typing import Any, Dict, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.mmst.constants import (
    DUMMY_NODE,
    EMPTY_NODE,
    INVALID_ALREADY_TRAVERSED,
    INVALID_CHOICE,
    INVALID_NODE,
    INVALID_TIE_BREAK,
    UTILITY_NODE,
)
from jumanji.environments.routing.mmst.generator import Generator, SplitRandomGenerator
from jumanji.environments.routing.mmst.reward import DenseRewardFn, RewardFn
from jumanji.environments.routing.mmst.types import Observation, State
from jumanji.environments.routing.mmst.utils import (
    make_action_mask,
    update_active_edges,
)
from jumanji.environments.routing.mmst.viewer import MMSTViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class MMST(Environment[State]):
    """The `MMST` (Multi Minimum Spanning Tree) environment
    consists of a random connected graph
    with groups of nodes (same node types) that needs to be connected.
    The goal of the environment is to connect all nodes of the same type together
    without using the same utility nodes (nodes that do not belong to any group of nodes).

    Note: routing problems are randomly generated and may not be solvable!

    Requirements: The total number of nodes should be at least 20% more than
    the number of nodes we want to connect to guarantee we have enough remaining
    nodes to create a path with all the nodes we want to connect.
    An exception will be raised if the number of nodes is not greater
    than (0.8 x num_agents x num_nodes_per_agent).

    - observation: Observation
        - node_types: jax array (int) of shape (num_nodes):
            the component type of each node (-1 represents utility nodes).
        - adj_matrix: jax array (bool) of shape (num_nodes, num_nodes):
            adjacency matrix of the graph.
        - positions: jax array (int) of shape (num_agents,):
            the index of the last visited node.
        - step_count: jax array (int) of shape ():
            integer to keep track of the number of steps.
        - action_mask: jax array (bool) of shape (num_agent, num_nodes):
            binary mask (False/True <--> invalid/valid action).

    - reward: float

    - action: jax array (int) of shape (num_agents,): [0,1,..., num_nodes-1]
        Each agent selects the next node to which it wants to connect.

    - state: State
        - node_type: jax array (int) of shape (num_nodes,).
            the component type of each node (-1 represents utility nodes).
        - adj_matrix: jax array (bool) of shape (num_nodes, num_nodes):
            adjacency matrix of the graph.
        - connected_nodes: jax array (int) of shape (num_agents, time_limit).
            we only count each node visit once.
        - connected_nodes_index: jax array (int) of shape (num_agents, num_nodes).
        - position_index: jax array (int) of shape (num_agents,).
        - node_edges: jax array (int) of shape (num_agents, num_nodes, num_nodes).
        - positions: jax array (int) of shape (num_agents,).
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

        - Actions encoding
            - INVALID_CHOICE = -1
            - INVALID_TIE_BREAK = -2
            - INVALID_ALREADY_TRAVERSED = -3
    """

    def __init__(
        self,
        generator: Optional[Generator] = None,
        reward_fn: Optional[RewardFn] = None,
        time_limit: int = 70,
        viewer: Optional[Viewer[State]] = None,
    ):
        """Create the `MMST` environment.

        Args:
            generator: `Generator` whose `__call__` instantiates an environment instance.
                Implemented options are [`SplitRandomGenerator`].
                Defaults to `SplitRandomGenerator(num_nodes=36, num_edges=72, max_degree=5,
                num_agents=3, num_nodes_per_agent=4, max_step=time_limit)`.
            reward_fn: class of type `RewardFn`, whose `__call__` is used as a reward function.
                Implemented options are [`DenseRewardFn`].
                Defaults to `DenseRewardFn(reward_values=(10.0, -1.0, -1.0))`.
            time_limit: the number of steps allowed before an episode terminates. Defaults to 70.
            viewer: `Viewer` used for rendering. Defaults to `MMSTViewer`
        """

        self._generator = generator or SplitRandomGenerator(
            num_nodes=36,
            num_edges=72,
            max_degree=5,
            num_agents=3,
            num_nodes_per_agent=4,
            max_step=time_limit,
        )

        self.num_agents = self._generator.num_agents
        self.num_nodes = self._generator.num_nodes
        self.num_nodes_per_agent = self._generator.num_nodes_per_agent

        self._reward_fn = reward_fn or DenseRewardFn(reward_values=(10.0, -1.0, -1.0))

        self._env_viewer = viewer or MMSTViewer(num_agents=self.num_agents)
        self.time_limit = time_limit

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment.

        Args:
            key: used to randomly generate the problem and the different start nodes.

        Returns:
             state: State object corresponding to the new state of the environment.
             timestep: TimeStep object corresponding to the first timestep returned by the
                environment.
        """

        key, problem_key = jax.random.split(key)
        state = self._generator(problem_key)
        extras = self._get_extras(state)
        timestep = restart(observation=self._state_to_observation(state), extras=extras)
        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the index of the next node to visit.

        Returns:
            state, timestep: Tuple[State, TimeStep] containing the next state of the
               environment, as well as the timestep to be observed.
        """

        def step_agent_fn(
            connected_nodes: chex.Array,
            conn_index: chex.Array,
            action: chex.Array,
            node: int,
            indices: chex.Array,
            agent_id: int,
        ) -> Tuple[chex.Array, ...]:

            is_invalid_choice = jnp.any(action == INVALID_CHOICE) | jnp.any(
                action == INVALID_TIE_BREAK
            )
            is_valid = (~is_invalid_choice) & (node != INVALID_NODE)
            connected_nodes, conn_index, new_node, indices = jax.lax.cond(
                is_valid,
                self._update_conected_nodes,
                lambda *_: (
                    connected_nodes,
                    conn_index,
                    state.positions[agent_id],
                    indices,
                ),
                connected_nodes,
                conn_index,
                node,
                indices,
            )

            return connected_nodes, conn_index, new_node, indices

        key, step_key = jax.random.split(state.key)
        action, next_nodes = self._trim_duplicated_invalid_actions(
            state, action, step_key
        )

        connected_nodes = jnp.zeros_like(state.connected_nodes)
        connected_nodes_index = jnp.zeros_like(state.connected_nodes_index)
        agents_pos = jnp.zeros_like(state.positions)
        position_index = jnp.zeros_like(state.position_index)

        for agent in range(self.num_agents):
            conn_nodes_i, conn_nodes_id, pos_i, pos_ind = step_agent_fn(
                state.connected_nodes[agent],
                state.connected_nodes_index[agent],
                action[agent],
                next_nodes[agent],
                state.position_index[agent],
                agent,
            )

            connected_nodes = connected_nodes.at[agent].set(conn_nodes_i)
            connected_nodes_index = connected_nodes_index.at[agent].set(conn_nodes_id)
            agents_pos = agents_pos.at[agent].set(pos_i)
            position_index = position_index.at[agent].set(pos_ind)

        active_node_edges = update_active_edges(
            self.num_agents, state.node_edges, agents_pos, state.node_types
        )

        state = State(
            node_types=state.node_types,
            adj_matrix=state.adj_matrix,
            nodes_to_connect=state.nodes_to_connect,
            connected_nodes=connected_nodes,
            connected_nodes_index=connected_nodes_index,
            position_index=position_index,
            positions=agents_pos,
            node_edges=active_node_edges,
            action_mask=make_action_mask(
                self.num_agents,
                self.num_nodes,
                active_node_edges,
                agents_pos,
                state.finished_agents,
            ),
            finished_agents=state.finished_agents,  # Not updated yet.
            step_count=state.step_count,
            key=key,
        )

        state, timestep = self._state_to_timestep(state, action)
        return state, timestep

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec.

        Returns:
            action_spec: a `specs.MultiDiscreteArray` spec.
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.full((self.num_agents,), self.num_nodes, jnp.int32),
            name="action",
        )

    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec.

        Returns:
            Spec for the `Observation` whose fields are:
            - node_types: BoundedArray (int32) of shape (num_nodes,).
            - adj_matrix: BoundedArray (int) of shape (num_nodes, num_nodes).
                Represents the adjacency matrix of the graph.
            - positions: BoundedArray (int32) of shape (num_agents).
                Current node position of agent.
            - action_mask: BoundedArray (bool) of shape (num_agents, num_nodes,).
                Represents the valid actions in the current state.
        """
        node_types = specs.BoundedArray(
            shape=(self.num_nodes,),
            minimum=-1,
            maximum=self.num_agents * 2 - 1,
            dtype=jnp.int32,
            name="node_types",
        )
        adj_matrix = specs.BoundedArray(
            shape=(self.num_nodes, self.num_nodes),
            minimum=0,
            maximum=1,
            dtype=jnp.int32,
            name="adj_matrix",
        )
        positions = specs.BoundedArray(
            shape=(self.num_agents,),
            minimum=-1,
            maximum=self.num_nodes - 1,
            dtype=jnp.int32,
            name="positions",
        )
        step_count = specs.BoundedArray(
            shape=(),
            minimum=0,
            maximum=self.time_limit,
            dtype=jnp.int32,
            name="step_count",
        )
        action_mask = specs.BoundedArray(
            shape=(self.num_agents, self.num_nodes),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask",
        )

        return specs.Spec(
            Observation,
            "ObservationSpec",
            node_types=node_types,
            adj_matrix=adj_matrix,
            positions=positions,
            step_count=step_count,
            action_mask=action_mask,
        )

    def _state_to_observation(self, state: State) -> Observation:
        """Converts a state into an observation.

        Args:
            state: State object containing the dynamics of the environment.

        Returns:
            observation: Observation object containing the observation of the environment.
        """

        # Each agent should see its note_types labelled with id 1
        # and all its already connected nodes labeled with id 0.

        # Preserve the negative ones.
        zero_mask = state.node_types != UTILITY_NODE
        ones_inds = state.node_types == UTILITY_NODE

        # Set the agent_id to 0 since the environment is single agent.
        agent_id = 0

        node_types = state.node_types - agent_id
        node_types %= self.num_agents
        node_types *= 2
        node_types += 1  # Add one so that current agent nodes are labelled 1.

        node_types *= zero_mask  # Mask the position with negative ones.
        node_types -= ones_inds  # Add the negative ones back.

        # Set already connected nodes by agent to 0.
        for agent in range(self.num_agents):
            connected_mask = state.connected_nodes_index[agent] == UTILITY_NODE
            connected_ones = state.connected_nodes_index[agent] != UTILITY_NODE
            node_types *= connected_mask
            agent_skip = (agent - agent_id) % self.num_agents
            node_types += (2 * agent_skip) * connected_ones

        return Observation(
            node_types=node_types,
            adj_matrix=state.adj_matrix,
            positions=state.positions,
            step_count=state.step_count,
            action_mask=state.action_mask,
        )

    def _state_to_timestep(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        """Checks if the state is terminal and converts it into a timestep.

        Args:
            state: State object containing the dynamics of the environment.
            action: action taken the agent in this step.

        Returns:
            timestep: TimeStep object containing the timestep of the environment.
        """

        reward = self._reward_fn(state, action, state.nodes_to_connect)

        # Update the state now.
        state.finished_agents = self.get_finished_agents(state)
        state.step_count = state.step_count + 1
        extras = self._get_extras(state)
        observation = self._state_to_observation(state)

        def make_termination_timestep() -> TimeStep[Observation]:
            return termination(
                reward=reward,
                observation=observation,
                extras=extras,
            )

        def make_transition_timestep() -> TimeStep[Observation]:
            return transition(
                reward=reward,
                observation=observation,
                extras=extras,
            )

        agents_are_done = state.finished_agents.all()
        horizon_reached = state.step_count >= self.time_limit

        timestep = jax.lax.cond(
            agents_are_done | horizon_reached,
            make_termination_timestep,
            make_transition_timestep,
        )

        return state, timestep

    def _trim_duplicated_invalid_actions(
        self, state: State, action: chex.Array, step_key: chex.PRNGKey
    ) -> chex.Array:
        """Check for duplicated actions and randomly break ties.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the index of the next node to visit.
        Returns:
            action: Array containing the index of the next node to visit.
                -2 indicates do not move because of tie break
                -1 indicates do not move because of an invalid choice
                -3 indicates moving to an already traversed node
            nodes: actual new nodes
                -1 invalid node no movement
        """

        def _get_agent_node(
            node_edges: chex.Array, position: chex.Array, action: chex.Array
        ) -> chex.Array:
            node = node_edges[position, action]
            return node

        nodes = jax.vmap(_get_agent_node)(state.node_edges, state.positions, action)

        new_actions = jnp.ones_like(action) * INVALID_CHOICE

        added_nodes = jnp.ones((self.num_agents), dtype=jnp.int32) * DUMMY_NODE

        agent_permutation = jax.random.permutation(
            step_key, jnp.arange(self.num_agents)
        )

        def not_all_agents_actions_examined(arg: Any) -> Any:
            added_nodes, new_actions, action, nodes, agent_permutation, index = arg
            return index < self.num_agents

        def modify_action_if_agent_target_node_is_selected(arg: Any) -> Any:
            added_nodes, new_actions, action, nodes, agent_permutation, index = arg
            agent_i = agent_permutation[index]

            is_invalid_node = nodes[agent_i] == EMPTY_NODE
            node_is_not_selected = jnp.sum(jnp.sum(added_nodes == nodes[agent_i]) == 0)

            # false + false = 0 = tie break.
            # true + false = 1  = invalid choice (with tie break) do nothing.
            # false + true * 2 = 2 = valid choice and valid node.
            # true + true * 2 = 3 -> invalid choice (without tie break).

            new_actions, added_nodes = jax.lax.switch(
                is_invalid_node + node_is_not_selected * 2,
                [
                    lambda *_: (
                        new_actions.at[agent_i].set(INVALID_TIE_BREAK),
                        added_nodes.at[agent_i].set(INVALID_TIE_BREAK),
                    ),
                    lambda *_: (new_actions, added_nodes),
                    lambda *_: (
                        new_actions.at[agent_i].set(action[agent_i]),
                        added_nodes.at[agent_i].set(nodes[agent_i]),
                    ),
                    lambda *_: (new_actions, added_nodes),
                ],
                new_actions,
                added_nodes,
            )
            index += 1

            return (added_nodes, new_actions, action, nodes, agent_permutation, index)

        (
            added_nodes,
            new_actions,
            action,
            nodes,
            agent_permutation,
            index,
        ) = jax.lax.while_loop(
            not_all_agents_actions_examined,
            modify_action_if_agent_target_node_is_selected,
            (added_nodes, new_actions, action, nodes, agent_permutation, 0),
        )

        def mask_visited_nodes(
            node_visited: jnp.int32, old_action: jnp.int32
        ) -> jnp.int32:
            new_action = jax.lax.cond(  # type:ignore
                node_visited != EMPTY_NODE,
                lambda *_: INVALID_ALREADY_TRAVERSED,
                lambda *_: old_action,
            )

            return new_action

        final_actions = jnp.zeros_like(new_actions)
        # Set the action to 0 if the agent is moving to an already connected node.
        for agent in range(self.num_agents):
            node_visited = state.connected_nodes_index[agent, nodes[agent]]
            new_action = mask_visited_nodes(node_visited, new_actions[agent])
            final_actions = final_actions.at[agent].set(new_action)

        # Mask agents with finished states.
        final_actions = final_actions * ~state.finished_agents - state.finished_agents

        return final_actions, nodes

    def _update_conected_nodes(
        self,
        connected_nodes: chex.Array,
        connected_node_index: chex.Array,
        node: int,
        index: int,
    ) -> chex.Array:
        """Add this node to the connected_nodes part of the specific agent

        Args:
            connected_nodes (Array): Nodes connected by each agent.
            connected_nodes_index (Array): Nodes connected by each agent.
            node (int): New node to connect
            index (int): position
        Returns:
            connected_nodes (Array): Array with connected node appended.
        """

        index += 1
        connected_nodes = connected_nodes.at[index].set(node)
        connected_node_index = connected_node_index.at[node].set(node)
        return connected_nodes, connected_node_index, node, index

    def get_finished_agents(self, state: State) -> chex.Array:
        """Get the done flags for each agent.

        Args:
            node_types: the environment state node_types.
            connected_nodes: the agent specifc view of connected nodes
        Returns:
            Array : array of boolean flags in the shape (number of agents, ).
        """

        def done_fun(
            nodes: chex.Array, connected_nodes: chex.Array, n_comps: int
        ) -> jnp.bool_:
            connects = jnp.isin(nodes, connected_nodes)
            return jnp.sum(connects) == n_comps

        finished_agents = jax.vmap(done_fun, in_axes=(0, 0, None))(
            state.nodes_to_connect,
            state.connected_nodes,
            self.num_nodes_per_agent,
        )

        return finished_agents

    def _get_extras(self, state: State) -> Dict:
        """Computes extras metrics to be return within the timestep."""

        def num_connections(
            nodes: chex.Array, connected_nodes: chex.Array, n_comps: int
        ) -> chex.Array:
            connects = jnp.isin(nodes, connected_nodes)
            total_connections = jnp.sum(connects) - 1.0
            ratio_connections = total_connections / (n_comps - 1.0)
            return jnp.array([total_connections, ratio_connections])

        connections = jax.vmap(num_connections, in_axes=(0, 0, None))(
            state.nodes_to_connect,
            state.connected_nodes,
            self.num_nodes_per_agent,
        )

        extras = {
            "num_connections": jnp.sum(connections[:, 0]),
            "ratio_connections": jnp.mean(connections[:, 1]),
        }

        return extras

    def render(self, state: State) -> chex.Array:
        """Render the environment for a given state.

        Returns:
            Array of rgb pixel values in the shape (width, height, rgb).
        """
        return self._env_viewer.render(state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Calls the environment renderer to animate a sequence of states.

        Args:
            states: List of states to animate.
            interval: Time between frames in milliseconds, defaults to 200.
            save_path: Optional path to save the animation.
        """
        return self._env_viewer.animate(states, interval, save_path)
