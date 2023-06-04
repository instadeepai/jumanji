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
import jax.numpy as jnp
import pytest

from jumanji.environments.routing.multi_cvrp.generator import UniformRandomGenerator
from jumanji.environments.routing.multi_cvrp.utils import DEPOT_IDX


@pytest.fixture
def uniform_random_generator() -> UniformRandomGenerator:
    """Creates a generator with 6 customers and 2 vehicles."""
    return UniformRandomGenerator(num_customers=6, num_vehicles=2)


@pytest.fixture
def key() -> chex.PRNGKey:
    """Generates a random key."""
    return jax.random.PRNGKey(0)


def test_uniform_random_generator__call(
    uniform_random_generator: UniformRandomGenerator, key: chex.PRNGKey
) -> None:
    """Test that generator generates valid state."""
    state = uniform_random_generator(key)

    # Check the state's fields have the expected shapes
    assert state.nodes.coordinates.shape == (6 + 1, 2)  # +1 for the depot
    assert state.nodes.demands.shape == (6 + 1,)
    assert state.windows.start.shape == (6 + 1,)
    assert state.windows.end.shape == (6 + 1,)
    assert state.coeffs.early.shape == (6 + 1,)
    assert state.coeffs.late.shape == (6 + 1,)
    assert state.vehicles.positions.shape == (2,)
    assert state.vehicles.local_times.shape == (2,)
    assert state.vehicles.capacities.shape == (2,)
    assert state.vehicles.distances.shape == (2,)
    assert state.vehicles.time_penalties.shape == (2,)
    assert state.order.shape == (2, 2 * 6)  # 2* because it's (customers + depot)
    assert state.step_count.shape == ()

    # Check vehicles start at the depot
    assert (state.vehicles.positions == jnp.array([DEPOT_IDX] * 2)).all()


def test_uniform_random_generator__no_retrace(
    uniform_random_generator: UniformRandomGenerator, key: chex.PRNGKey
) -> None:
    """Checks that generator only traces the function once and works when jitted."""
    keys = jax.random.split(key, 2)
    jitted_generator = jax.jit(
        chex.assert_max_traces((uniform_random_generator.__call__), n=1)
    )

    for key in keys:
        jitted_generator(key)
