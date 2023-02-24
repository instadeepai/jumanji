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

from typing import Optional, Tuple

import chex
import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.logic.planar.generator import Generator, RandomGenerator
from jumanji.environments.logic.planar.reward import (
    IntersectionCountChangeRewardFn,
    RewardFn,
)
from jumanji.environments.logic.planar.types import Observation, State
from jumanji.environments.logic.planar.viewer import Viewer, networkx_viewer
from jumanji.types import Action, TimeStep, restart, termination, transition


class PlanarGraph(Environment[State]):
    """A planar graph is a graph that can be drawn in a 2D plane such that its edges only intersect
    in their endpoints, i.e. they shouldn't cross each other if drawn as straight lines.
    The goal of this environment is to modify the positions of the graph nodes to make
    the graph planar. At each time step, all nodes can be moved in the 2D plane. An episode stops
    after a fixed number of steps defined by `time_limit`.
    """

    # Action for each node: No-op, Up, Right, Down, Left.
    _MOVES: chex.Array = jnp.asarray([[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]])

    def __init__(
        self,
        generator: Optional[Generator] = None,
        reward_fn: Optional[RewardFn] = None,
        viewer: Viewer = networkx_viewer,
        time_limit: int = 100,
    ):
        """Instantiates a PlanarGraph environment.

        Args:
            generator: callable to instantiate environment instances.
                Defaults to `RandomGenerator` with `num_nodes=10` and `num_edges=15`.
            reward_fn: callable to compute the reward of the environment's transition.
                Defaults to `IntersectionCountChangeRewardFn`.
            viewer: function to render an environment state. This used when calling `render`.
            time_limit: horizon of an episode, i.e. number of environment steps before
                the episode ends. Defaults to 100.
        """
        self.generator = generator or RandomGenerator(num_nodes=10, num_edges=15)
        self.reward_fn = reward_fn or IntersectionCountChangeRewardFn()
        self.viewer = viewer
        self.time_limit = time_limit

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        key, subkey = jax.random.split(key)
        nodes, edges = self.generator(subkey)

        # New state of the environment
        state = State(nodes=nodes, edges=edges, key=key, step=0)
        timestep = restart(
            observation=self.observation(state),
        )

        return state, timestep

    def step(self, state: State, action: Action) -> Tuple[State, TimeStep]:
        # Update each node position
        moves = jnp.take(PlanarGraph._MOVES, action, axis=0)
        nodes = state.nodes + moves

        # Create the new state
        next_state = State(
            key=state.key,
            nodes=nodes,
            edges=state.edges,
            step=state.step + 1,
        )

        # Compute the reward of the new state
        reward = self.reward_fn(state=next_state, prev_state=state)

        # Termination criteria
        done = self.termination(next_state)

        # Decide the step type to return
        timestep = jax.lax.cond(
            done,
            termination,
            transition,
            reward,
            self.observation(next_state),
        )

        return next_state, timestep

    def termination(self, state: State) -> jnp.bool_:
        return state.step == self.time_limit

    def observation_spec(self) -> specs.Spec:
        num_nodes, num_edges = self.generator.specs()
        return specs.Spec(
            Observation,
            "ObservationSpec",
            nodes=specs.Array(shape=(num_nodes, 2), dtype=jnp.int_),
            edges=specs.Array(shape=(num_edges, 2), dtype=jnp.int_),
        )

    def observation(self, state: State) -> Observation:
        return Observation(
            nodes=state.nodes,
            edges=state.edges,
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        num_actions = PlanarGraph._MOVES.shape[0]  # Number of actions per node
        num_nodes, _ = self.generator.specs()

        return specs.MultiDiscreteArray(
            num_values=jnp.array([num_actions] * num_nodes),
            name="action",
        )

    def reward_spec(self) -> specs.Array:
        return self.reward_fn.spec()

    def render(self, state: State) -> plt.Figure:
        return self.viewer(state)
