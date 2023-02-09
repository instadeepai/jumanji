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

import itertools as it
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import pytest

from jumanji.environments.logic.planar import generator


def get_graph_specs(graph: generator.Graph) -> Tuple[int, int]:
    nodes, edges = graph
    return nodes.shape[0], edges.shape[0]


class TestRandomGenerator:
    @pytest.mark.parametrize(("num_nodes", "num_edges"), ((5, 10), (10, 3)))
    def test_random__specs(self, num_nodes: int, num_edges: int) -> None:
        gen = generator.RandomGenerator(num_nodes=num_nodes, num_edges=num_edges)
        assert gen.specs() == (num_nodes, num_edges)
        assert gen.specs() == get_graph_specs(gen(jax.random.PRNGKey(0)))

    def test_random__too_many_edges(self) -> None:
        num_nodes = 10
        max_num_edges = num_nodes**2

        # Pass if you request the max number of edges
        gen = generator.RandomGenerator(num_nodes=num_nodes, num_edges=max_num_edges)

        # The call must pass as well
        _ = gen(jax.random.PRNGKey(0))

        # Fail if you exceed this number
        with pytest.raises(AssertionError):
            _ = generator.RandomGenerator(
                num_nodes=num_nodes, num_edges=max_num_edges + 1
            )

    def test_random__edges(self) -> None:
        num_nodes, num_edges, num_trials = 10, 25, 42

        keys = jax.random.split(jax.random.PRNGKey(123), num_trials)
        gen = generator.RandomGenerator(num_nodes=num_nodes, num_edges=num_edges)
        for key in keys:
            _, edges = gen(key)

            # No self-loop
            assert all(n1 != n2 for n1, n2 in edges)

            # No mirror edges
            # Note: the speed of this test could be improved by vmap-ing the comparison
            def _equal(args: Tuple) -> chex.Array:
                e1, e2 = args
                return jnp.array_equiv(e1, e2) or jnp.array_equiv(e1, jnp.flip(e2))

            num_equivalent = sum(map(_equal, it.product(edges, repeat=2)))
            assert num_equivalent == num_edges  # self-comparisons are legit


class TestChainGenerator:
    @pytest.mark.parametrize("length", (1, 2, 10))
    def test_chain__specs(self, length: int) -> None:
        num_nodes, num_edges = length, length - 1
        gen = generator.ChainGenerator(num_nodes=length)
        assert gen.specs() == (num_nodes, num_edges)
        assert gen.specs() == get_graph_specs(gen(jax.random.PRNGKey(0)))


class TestDummyGenerator:
    def test_chain__specs(self) -> None:
        gen = generator.DummyGenerator()
        assert gen.specs() == get_graph_specs(gen(jax.random.PRNGKey(0)))
