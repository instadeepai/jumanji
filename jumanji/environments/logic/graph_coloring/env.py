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

from typing import Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib.animation as animation
from jax import lax
from numpy.typing import NDArray

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.logic.graph_coloring.generator import (
    Generator,
    RandomGenerator,
)
from jumanji.environments.logic.graph_coloring.types import Observation, State
from jumanji.environments.logic.graph_coloring.viewer import GraphColoringViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class GraphColoring(Environment[State]):
    """Environment for the GraphColoring problem.
    The problem is a combinatorial optimization task where the goal is
      to assign a color to each vertex of a graph
      in such a way that no two adjacent vertices share the same color.
    The problem is usually formulated as minimizing the number of colors used.

    - observation: `Observation`
        - adj_matrix: jax array (bool) of shape (num_nodes, num_nodes),
            representing the adjacency matrix of the graph.
        - colors: jax array (int32) of shape (num_nodes,),
            representing the current color assignments for the vertices.
        - action_mask: jax array (bool) of shape (num_colors,),
            indicating which actions are valid in the current state of the environment.
        - current_node_index: integer representing the current node being colored.

    - action: int, the color to be assigned to the current node (0 to num_nodes - 1)

     - reward: float, a sparse reward is provided at the end of the episode.
        Equals the negative of the number of unique colors used to color all vertices in the graph.
        If an invalid action is taken, the reward is the negative of the total number of colors.

    - episode termination:
        - if all nodes have been assigned a color or if an invalid action is taken.

    - state: `State`
        - adj_matrix: jax array (bool) of shape (num_nodes, num_nodes),
            representing the adjacency matrix of the graph.
        - colors: jax array (int32) of shape (num_nodes,),
            color assigned to each node, -1 if not assigned.
        - current_node_index: jax array (int) with shape (),
            index of the current node.
        - action_mask: jax array (bool) of shape (num_colors,),
            indicating which actions are valid in the current state of the environment.
        - key: jax array (uint32) of shape (2,),
            random key used to generate random numbers at each step and for auto-reset.

    ```python
    from jumanji.environments import GraphColoring
    env = GraphColoring()
    key = jax.random.key(0)
    state, timestep = jax.jit(env.reset)(key)
    env.render(state)
    action = env.action_spec().generate_value()
    state, timestep = jax.jit(env.step)(state, action)
    env.render(state)
    ```
    """

    def __init__(
        self,
        generator: Optional[Generator] = None,
        viewer: Optional[Viewer[State]] = None,
    ):
        """Instantiate a `GraphColoring` environment.

        Args:
            generator: callable to instantiate environment instances.
                Defaults to `RandomGenerator` which generates graphs with
                20 `num_nodes` and `edge_probability` equal to 0.8.
            viewer: environment viewer for rendering.
                Defaults to `GraphColoringViewer`.
        """
        self.generator = generator or RandomGenerator(
            num_nodes=20, edge_probability=0.8
        )
        self.num_nodes = self.generator.num_nodes

        # Create viewer used for rendering
        self._env_viewer = viewer or GraphColoringViewer(name="GraphColoring")

    def __repr__(self) -> str:
        return repr(self.generator)

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment to an initial state.

        Returns:
            The initial state and timestep.
        """
        colors = jnp.full(self.num_nodes, -1, dtype=jnp.int32)
        key, subkey = jax.random.split(key)
        adj_matrix = self.generator(subkey)

        action_mask = jnp.ones(self.num_nodes, dtype=bool)
        current_node_index = jnp.array(0, jnp.int32)
        state = State(
            adj_matrix=adj_matrix,
            colors=colors,
            current_node_index=current_node_index,
            action_mask=action_mask,
            key=key,
        )
        obs = Observation(
            adj_matrix=adj_matrix,
            colors=colors,
            action_mask=action_mask,
            current_node_index=current_node_index,
        )
        timestep = restart(observation=obs)

        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        """Updates the environment state after the agent takes an action.

        Specifically, this function allows the agent to choose
        a color for the current node (based on the action taken)
        in a graph coloring problem.
        It then updates the state of the environment based on
        the color chosen and calculates the reward based on
        the validity of the action and the completion of the coloring task.

        Args:
            state: the current state of the environment.
            action: the action taken by the agent.

        Returns:
            state: the new state of the environment.
            timestep: the next timestep.
        """
        # Get the valid actions for the current state.
        valid_actions = state.action_mask

        # Check if the chosen action is invalid (not in valid_actions).
        invalid_action_taken = jnp.logical_not(valid_actions[action])

        # Update the colors array with the chosen action.
        colors = state.colors.at[state.current_node_index].set(action)

        # Determine if all nodes have been assigned a color
        all_nodes_colored = jnp.all(colors >= 0)

        # Calculate the reward
        unique_colors_used = jnp.unique(colors, size=self.num_nodes, fill_value=-1)
        num_unique_colors = jnp.count_nonzero(unique_colors_used >= 0)
        reward = jnp.where(all_nodes_colored, -num_unique_colors, 0.0)

        # Apply the maximum penalty when an invalid action is taken and terminate the episode
        reward = jnp.where(invalid_action_taken, -self.num_nodes, reward)
        done = jnp.logical_or(all_nodes_colored, invalid_action_taken)

        # Update the current node index
        next_node_index = (state.current_node_index + 1) % self.num_nodes

        next_action_mask = self._get_valid_actions(
            next_node_index, state.adj_matrix, state.colors
        )

        next_state = State(
            adj_matrix=state.adj_matrix,
            colors=colors,
            current_node_index=next_node_index,
            action_mask=next_action_mask,
            key=state.key,
        )
        obs = Observation(
            adj_matrix=state.adj_matrix,
            colors=colors,
            action_mask=next_state.action_mask,
            current_node_index=next_node_index,
        )
        timestep = lax.cond(
            done,
            termination,
            transition,
            reward,
            obs,
        )
        return next_state, timestep

    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec.

        Returns:
            Spec for the `Observation` whose fields are:
            - adj_matrix: BoundedArray (bool) of shape (num_nodes, num_nodes).
                Represents the adjacency matrix of the graph.
            - action_mask: BoundedArray (bool) of shape (num_nodes,).
                Represents the valid actions in the current state.
            - colors: BoundedArray (int32) of shape (num_nodes,).
                Represents the colors assigned to each node.
            - current_node_index: BoundedArray (int32) of shape ().
                Represents the index of the current node.
        """
        return specs.Spec(
            Observation,
            "ObservationSpec",
            adj_matrix=specs.BoundedArray(
                shape=(self.num_nodes, self.num_nodes),
                dtype=bool,
                minimum=False,
                maximum=True,
                name="adj_matrix",
            ),
            action_mask=specs.BoundedArray(
                shape=(self.num_nodes,),
                dtype=bool,
                minimum=False,
                maximum=True,
                name="action_mask",
            ),
            colors=specs.BoundedArray(
                shape=(self.num_nodes,),
                dtype=jnp.int32,
                minimum=-1,
                maximum=self.num_nodes - 1,
                name="colors",
            ),
            current_node_index=specs.BoundedArray(
                shape=(),
                dtype=jnp.int32,
                minimum=0,
                maximum=self.num_nodes - 1,
                name="current_node_index",
            ),
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Specification of the action for the `GraphColoring` environment.

        Returns:
            action_spec: specs.DiscreteArray object
        """
        return specs.DiscreteArray(
            num_values=self.num_nodes, name="action", dtype=jnp.int32
        )

    def _get_valid_actions(
        self, current_node_index: int, adj_matrix: chex.Array, colors: chex.Array
    ) -> chex.Array:
        """Returns a boolean array indicating the valid colors for the current node."""
        # Create a boolean array of size (num_nodes + 1) set to True.
        # The extra element is to accommodate for the -1 index
        # which represents nodes that have not been colored yet.
        valid_actions = jnp.ones(self.num_nodes + 1, dtype=bool)
        row = adj_matrix[current_node_index, :]
        action_mask = jnp.where(row, colors, -1)
        valid_actions = valid_actions.at[action_mask].set(False)

        # Exclude the last element (which corresponds to -1 index)
        return valid_actions[:-1]

    def render(self, state: State) -> Optional[NDArray]:
        """Renders the current state of the `GraphColoring` environment.

        Args:
            state: is the current game state to be rendered.
        """
        return self._env_viewer.render(state=state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> animation.FuncAnimation:
        """Creates an animated gif of the `GraphColoring` environment based on the sequence of game states.

        Args:
            states: is a list of `State` objects representing the sequence of game states.
            interval: the delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be stored.

        Returns:
            animation.FuncAnimation: the animation object that was created.
        """
        return self._env_viewer.animate(
            states=states, interval=interval, save_path=save_path
        )

    def close(self) -> None:
        """Perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        self._env_viewer.close()
