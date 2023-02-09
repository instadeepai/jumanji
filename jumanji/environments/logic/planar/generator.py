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
import abc
from typing import Tuple

import chex
import jax
from jax import numpy as jnp

Graph = Tuple[chex.Array, chex.Array]


class Generator(abc.ABC):
    @abc.abstractmethod
    def specs(self) -> Tuple[int, int]:
        """Specs of the problem instances generated.

        Returns:
            a tuple with `num_nodes` and `num_edges`.
        """

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> Graph:
        """Generate a problem instance.

        Args:
            key: random key.

        Returns:
            A `graph` representing a problem instance.
        """


class RandomGenerator(Generator):
    """Generator of random graphs containing `num_nodes` and `num_edges`.

    The edges are:
        - Undirected, i.e. a -- b is equal to b -- a.
        - Unique, i.e. sampling without replacement.
        - Loop-less, i.e. if there exists an edge a -- b, then nodes a and b are different.

    Remark:
        There is no guarantee that the generated graph will be fully connected and/or solvable.
        Multiple nodes can share the same position when generated.
    """

    def __init__(self, num_nodes: int, num_edges: int, range_: int = 10):
        """

        Args:
            num_nodes: the number of nodes of the generated instances.
            num_edges: the number of edges of the generated instances.
            range_: nodes' positions will be sampled in the space (-range_, range_).
        """
        assert (
            num_edges <= num_nodes**2
        ), f"num_edges={num_edges} exceeds the number of edges of a fully-connected graph."

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.range_ = range_

    def specs(self) -> Tuple[int, int]:
        return self.num_nodes, self.num_edges

    def __call__(self, key: chex.PRNGKey) -> Graph:
        """Generate a random graph.

        Args:
            key: random key (consumed).

        Returns:
            A `Graph` containing:
            - The 2D positions of the nodes, i32[`num_nodes`, 2]. All positions
              are uniformly sampled in the range (-`range_`, +`range_`).
            - Array of edges connecting the nodes, i32[`num_edges`, 2].
        """
        k1, k2 = jax.random.split(key)

        # Initial nodes' position
        nodes = jax.random.randint(
            k1, shape=(self.num_nodes, 2), minval=-self.range_, maxval=self.range_
        )

        # Generate unique undirected random edges
        edges = jnp.stack(jnp.triu_indices(self.num_nodes, k=1), axis=1)
        edges = jax.random.permutation(k2, edges, axis=0)
        edges = edges[: self.num_edges]

        return nodes, edges


class ChainGenerator(Generator):
    """Generator of chain graphs with `num_nodes`, i.e. (1)--(2)--...--(`num_nodes`).

    The position of the nodes is initialized at random. The topology of the graph
    guarantees there exists a solution.

    Multiple nodes can share the same position when generated.
    """

    def __init__(self, num_nodes: int, range_: int = 10):
        self.num_nodes = num_nodes
        self.range_ = range_

    def specs(self) -> Tuple[int, int]:
        return self.num_nodes, self.num_nodes - 1

    def __call__(self, key: chex.PRNGKey) -> Graph:
        """Generate a chain graph.

        Args:
            key: random key (consumed).

        Returns:
            - The 2D positions of the nodes, i32[`num_nodes`, 2].
            - Array of edges connecting the nodes, i32[`num_edges`, 2].
        """

        nodes = jax.random.randint(
            key, shape=(self.num_nodes, 2), minval=-self.range_, maxval=self.range_
        )

        edges = jnp.stack(
            (jnp.arange(0, self.num_nodes - 1), jnp.arange(1, self.num_nodes)), axis=-1
        )

        return nodes, edges


class DummyGenerator(Generator):
    """Generator of a hardcoded solvable instance containing 4 nodes, 3 edges
    and having a single crossing when initialised.
    """

    def __init__(self) -> None:
        self.num_nodes = 4
        self.num_edges = 3
        self.range_ = 10

    def specs(self) -> Tuple[int, int]:
        return self.num_nodes, self.num_edges

    def __call__(self, key: chex.PRNGKey) -> Graph:
        """Generate a dummy hardcoded problem instance.

        Args:
            key: random key (not consumed).
        """
        x = self.range_ // 2
        nodes = jnp.asarray([[-x, x], [x, -x], [x, x], [-x, -x]], dtype=jnp.int32)

        # chain graph
        edges = jnp.stack(
            [jnp.arange(0, self.num_nodes - 1), jnp.arange(1, self.num_nodes)], axis=-1
        )

        return nodes, edges
