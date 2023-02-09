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
import jax
import pytest
from jax import numpy as jnp

from jumanji.environments.logic.planar import reward
from jumanji.environments.logic.planar.generator import RandomGenerator
from jumanji.environments.logic.planar.types import State

NODES = jnp.asarray([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=jnp.int32)


@pytest.fixture
def fake_state() -> State:
    """Returns a fake dummy state containing a random graph with 5 nodes and 7 edges"""
    key = jax.random.PRNGKey(0)
    nodes, edges = RandomGenerator(num_nodes=5, num_edges=7)(key)
    return State(nodes=nodes, edges=edges, key=key, step=0)


class TestIntersectionCount:
    def test_intersection_count__spec(self, fake_state: State) -> None:
        reward_fn = reward.IntersectionCountRewardFn()

        out = reward_fn(fake_state, fake_state)  # unused 2nd argument
        chex.assert_shape(out, reward_fn.spec().shape)
        chex.assert_type(out, reward_fn.spec().dtype)  # type: ignore

    @pytest.mark.parametrize(
        ("edges", "expected"),
        (
            ([[0, 1]], 0),
            ([[0, 1], [1, 0]], 0),
            ([[0, 1], [2, 3]], 1),
            ([[0, 1], [2, 3], [3, 0]], 1),
        ),
        ids=(
            "no crossing",
            "mirror edges",
            "1 cross with 2 edges",
            "1 cross with 3 edges",
        ),
    )
    def test_intersection_count__value(self, edges: List, expected: int) -> None:
        edges = jnp.asarray(edges, dtype=jnp.int32)
        state = State(nodes=NODES, edges=edges, key=jax.random.PRNGKey(0), step=0)
        out = reward.IntersectionCountRewardFn()(state, state)  # unused 2nd argument
        chex.assert_equal(-expected, out)


class TestIntersectionCountChange:
    def test_intersection_count_change__spec(self, fake_state: State) -> None:
        reward_fn = reward.IntersectionCountChangeRewardFn()

        out = reward_fn(fake_state, fake_state)
        chex.assert_shape(out, reward_fn.spec().shape)
        chex.assert_type(out, reward_fn.spec().dtype)  # type: ignore

    def test_intersection_count_change__value(self) -> None:
        reward_fn = reward.IntersectionCountChangeRewardFn()

        # Initial state
        nodes = jnp.asarray([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=jnp.int32)
        edges = jnp.asarray([[0, 1], [1, 2], [3, 0]], dtype=jnp.int32)
        state = State(key=jax.random.PRNGKey(0), nodes=nodes, edges=edges, step=0)

        num_nodes = nodes.shape[0]
        prefix = jnp.zeros((num_nodes - 1, 2), dtype=jnp.int32)

        # If the nodes remain unchanged, reward is zero.
        chex.assert_equal(0, reward_fn(state, state))

        # Move the last node such that it creates an intersection
        move = jnp.concatenate((prefix, jnp.asarray([[0, 1]])), axis=0)
        prev_state, state = state, state.replace(nodes=state.nodes + move)  # type: ignore
        chex.assert_equal(-1, reward_fn(state, prev_state))

        # Keep moving the last node.
        # the previous intersection remains and a new one is created.
        # Hence, the effective reward is minus one.
        move = jnp.concatenate((prefix, jnp.asarray([[-1, 0]])), axis=0)
        prev_state, state = state, state.replace(nodes=state.nodes + move)
        chex.assert_equal(-1, reward_fn(state, prev_state))

        # Keep moving the last node.
        # The previous intersections disappear, hence the reward is +2.
        move = jnp.concatenate((prefix, jnp.asarray([[-1, 0]])), axis=0)
        prev_state, state = state, state.replace(nodes=state.nodes + move)
        chex.assert_equal(2, reward_fn(state, prev_state))
