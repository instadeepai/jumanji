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
                Defaults to `RandomGenerator` with `num_nodes = 100`,
                and `percent_connected = 0.8` parameters.
        """
        self.generator = generator or RandomGenerator(
            num_nodes=100,
            percent_connected=0.8,
        )
        num_nodes, percent_connected = self.generator.specs()

        # Create viewer used for rendering
        self._env_viewer = viewer or GraphColoringViewer(
            num_nodes=num_nodes, name="GraphColoring"
        )

    def __repr__(self) -> str:
        """Returns: str: the string representation of the environment."""
        num_nodes, percent_connected = self.generator.specs()
        return (
            f"GraphColoring(number of nodes={num_nodes}, "
            f"percent connected={percent_connected * 100}% "
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment to an initial state.

        Returns:
            the initial state and timestep.
        """
        num_nodes, percent_connected = self.generator.specs()
        colors = jnp.full(num_nodes, -1, dtype=jnp.int32)
        key, subkey = jax.random.split(key)
        adj_matrix = self.generator(subkey)

        action_mask = jnp.ones(num_nodes, dtype=bool)
        state = State(
            adj_matrix=adj_matrix,
            colors=colors,
            current_node_index=jnp.array(0),
            key=key,
        )
        obs = Observation(
            adj_matrix=adj_matrix,
            colors=colors,
            action_mask=action_mask,
            current_node_index=jnp.array(0),
        )
        timestep = restart(observation=obs)

        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        """Updates the environment state after the agent takes an action.

        Args:
            state: the current state of the environment.
            action: the action taken by the agent.

        Returns:
            state: the new state of the environment.
            timestep: the next timestep.
        """
        num_nodes, _ = self.generator.specs()
        # Get the valid actions for the current state.
        valid_actions = self._get_valid_actions(state)
        # Check if the chosen action is invalid (not in valid_actions).
        invalid_action_taken = jnp.logical_not(valid_actions[action])

        # Update the colors array with the chosen action.
        colors = state.colors.at[state.current_node_index].set(action)

        # Determine if all nodes have been assigned a color
        all_nodes_colored = jnp.all(colors >= 0)

        # Calculate the reward
        unique_colors_used = jnp.unique(colors, size=num_nodes, fill_value=-1)
        num_unique_colors = jnp.count_nonzero(unique_colors_used >= 0)
        reward = jnp.where(all_nodes_colored, -num_unique_colors, 0.0).sum()

        # Apply the maximum penalty when an invalid action is taken and terminate the episode
        reward = jnp.where(invalid_action_taken, -num_nodes, reward)
        done = jnp.logical_or(all_nodes_colored, invalid_action_taken)

        # Update the current node index
        next_node_index = (state.current_node_index + 1) % num_nodes

        next_state = State(
            adj_matrix=state.adj_matrix,
            colors=colors,
            current_node_index=next_node_index,
            key=state.key,
        )
        obs = Observation(
            adj_matrix=state.adj_matrix,
            colors=colors,
            action_mask=self._get_valid_actions(next_state),
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
        """Returns the observation spec containing the graph, colors, and current node index.

        Returns:
            observation_spec:
                - ObservationSpec tree of the graph, colors, and current node index spec.
        """
        num_nodes, percent_connected = self.generator.specs()
        return specs.Spec(
            Observation,
            "ObservationSpec",
            adj_matrix=specs.BoundedArray(
                shape=(num_nodes, num_nodes),
                dtype=bool,
                minimum=False,
                maximum=True,
                name="adj_matrix",
            ),
            action_mask=specs.BoundedArray(
                shape=(num_nodes,),
                dtype=bool,
                minimum=False,
                maximum=True,
                name="action_mask",
            ),
            colors=specs.BoundedArray(
                shape=(num_nodes,),
                dtype=jnp.int32,
                minimum=-1,
                maximum=num_nodes - 1,
                name="colors",
            ),
            current_node_index=specs.BoundedArray(
                shape=(),
                dtype=jnp.int32,
                minimum=0,
                maximum=num_nodes - 1,
                name="current_node_index",
            ),
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec. size of number of nodes & a value of 0.

        Returns:
            action_spec: specs.DiscreteArray object
        """
        num_nodes, percent_connected = self.generator.specs()
        return specs.DiscreteArray(num_values=num_nodes, name="action", dtype=jnp.int32)

    def _get_valid_actions(self, state: State) -> chex.Array:
        """Returns a boolean array indicating the valid colors for the current node."""
        num_nodes, percent_connected = self.generator.specs()
        valid_actions = jnp.ones(num_nodes + 1, dtype=bool)
        row = state.adj_matrix[state.current_node_index, :]
        action_mask = jnp.where(row, state.colors, -1)
        valid_actions = valid_actions.at[action_mask].set(False)
        return valid_actions[:-1]

    def render(self, state: State) -> Optional[NDArray]:
        """Renders the current state of the GraphColoring.

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
        """Creates an animated gif of the GraphColoring based on the sequence of game states.

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
