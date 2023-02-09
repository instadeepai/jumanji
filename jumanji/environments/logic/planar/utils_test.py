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

from typing import List

import chex
import pytest
from jax import numpy as jnp

from jumanji.environments.logic.planar import utils


class TestCheckIntersect:
    _NODES = jnp.asarray(
        [[0, 0], [1, 1], [1, 0], [0, 1], [2, 1], [1, 1], [3, 0]], dtype=jnp.int32
    )

    @pytest.mark.parametrize(
        ("e1", "e2", "do_cross"),
        (
            ([0, 1], [0, 1], False),
            ([0, 1], [1, 0], False),
            ([0, 1], [2, 4], False),
            ([0, 1], [2, 3], True),
            ([0, 1], [1, 2], False),
            ([0, 1], [5, 2], True),
            ([0, 1], [4, 6], False),
        ),
        ids=(
            "same edge",
            "same reversed edge",
            "parallel",
            "cross in the middle",
            "share a node",
            "share a position",
            "out-of-range crossing",
        ),
    )
    def test_check_intersect__crossing(
        self, e1: List, e2: List, do_cross: bool
    ) -> None:
        e1, e2 = jnp.asarray(e1, dtype=jnp.int32), jnp.asarray(e2, dtype=jnp.int32)
        s1, s2 = jnp.take(self._NODES, jnp.stack((e1, e2), axis=0), axis=0)
        chex.assert_equal(do_cross, utils.check_intersect(s1, s2, e1, e2))


@pytest.mark.parametrize("num_edges", (1, 3))
def test_intersection_map__shapes(num_edges: int) -> None:
    segments = jnp.ones((num_edges, 2, 2), dtype=jnp.int32)
    edges = jnp.ones((num_edges, 2), dtype=jnp.int32)
    chex.assert_shape(utils.intersection_map(segments, edges), (num_edges, num_edges))


@pytest.mark.parametrize(
    ("nodes", "edges", "count"),
    (
        ([[0, 0], [1, 1]], [[0, 1]], 0),
        ([[0, 0], [1, 1]], [[0, 1], [1, 0]], 0),
        ([[0, 0], [1, 1], [1, 0], [0, 1]], [[0, 1], [2, 3]], 1),
        ([[0, 0], [1, 1], [1, 0], [0, 1]], [[0, 1], [2, 3], [3, 0]], 1),
    ),
    ids=(
        "one edge",
        "mirror edges",
        "1 cross with 2 edges",
        "1 cross with 3 edges and shared nodes",
    ),
)
def test_intersection_count__value(nodes: List, edges: List, count: int) -> None:
    nodes = jnp.asarray(nodes, dtype=jnp.int32)
    edges = jnp.asarray(edges, dtype=jnp.int32)
    segments = jnp.take(nodes, edges, axis=0)
    chex.assert_equal(count, utils.intersection_count(segments, edges))
