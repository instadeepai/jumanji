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

import chex
import jax
import pytest
from jax import numpy as jnp

from jumanji.environments.logic.planar import generator, reward
from jumanji.environments.logic.planar.env import PlanarGraph
from jumanji.environments.logic.planar.types import State
from jumanji.specs import MultiDiscreteArray
from jumanji.types import TimeStep

TIME_LIMIT = 10


@pytest.fixture(scope="module")
def env() -> PlanarGraph:
    """Instantiates a default PlanarGraph environment."""
    return PlanarGraph(
        generator=generator.RandomGenerator(num_nodes=5, num_edges=7),
        reward_fn=reward.IntersectionCountRewardFn(),
        time_limit=TIME_LIMIT,
    )


class TestPlanarGraph:
    @pytest.mark.parametrize("patch_jit", (True, False))
    def test_planar_graph__reset(self, env: PlanarGraph, patch_jit: bool) -> None:
        with chex.fake_jit(enable_patching=patch_jit):
            state, timestep = jax.jit(env.reset)(jax.random.PRNGKey(0))
            assert isinstance(state, State)
            assert isinstance(timestep, TimeStep)
            assert state.step == 0

    @pytest.mark.parametrize("patch_jit", (True, False))
    def test_planar_graph__step(self, env: PlanarGraph, patch_jit: bool) -> None:
        with chex.fake_jit(enable_patching=patch_jit):
            key = jax.random.PRNGKey(0)
            action_spec: MultiDiscreteArray = env.action_spec()

            key, subkey = jax.random.split(key)
            state, timestep = jax.jit(env.reset)(subkey)

            # Randomly select a non-(no-op) action for each node
            key, subkey = jax.random.split(key)
            action = jax.random.randint(
                key=subkey, shape=action_spec.num_values.shape, minval=1, maxval=5
            )

            next_state, _ = jax.jit(env.step)(state, action)

            # Check the nodes have moved
            # Remark: a node has 2 coordinates, any(axis=-1) checks at least one of them changed.
            assert jnp.not_equal(next_state.nodes, state.nodes).any(axis=-1).all()

    def test_planar_graph__termination(self, env: PlanarGraph) -> None:
        key = jax.random.PRNGKey(0)
        action_spec: MultiDiscreteArray = env.action_spec()

        key, subkey = jax.random.split(key)
        state, timestep = env.reset(subkey)

        # Assert the step function is only traced once
        step_fn = jax.jit(chex.assert_max_traces(env.step, n=1))

        # RandomGenerator action till termination
        while not timestep.last():
            key, subkey = jax.random.split(key)
            action = jax.random.randint(
                key=subkey, shape=action_spec.num_values.shape, minval=0, maxval=5
            )
            state, timestep = step_fn(state, action)

        assert state.step == TIME_LIMIT
