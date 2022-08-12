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
from jax import random

from jumanji import specs
from jumanji.binpack.env import BinPack
from jumanji.binpack.specs import EMSSpec, ItemSpec, ObservationSpec
from jumanji.binpack.types import State
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import TimeStep


def test_binpack__reset(binpack_env: BinPack) -> None:
    """Validates the jitted reset of the environment."""
    chex.clear_trace_counter()
    reset_fn = jax.jit(chex.assert_max_traces(binpack_env.reset, n=1))

    key = random.PRNGKey(0)
    state, timestep, _ = reset_fn(key)
    # Call again to check it does not compile twice.
    state, timestep, _ = reset_fn(key)
    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # reset function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(state)
    assert state.ems_mask.any()
    assert state.items_mask.any()
    assert jnp.any(state.action_mask)


def test_binpack__spec(binpack_env: BinPack) -> None:
    """Validates the observation and action spec of the BinPack environment."""
    observation_spec = binpack_env.observation_spec()
    assert isinstance(observation_spec, ObservationSpec)
    assert isinstance(observation_spec.ems_spec, EMSSpec)
    assert isinstance(observation_spec.ems_mask_spec, specs.BoundedArray)
    assert isinstance(observation_spec.items_spec, ItemSpec)
    assert isinstance(observation_spec.items_mask_spec, specs.BoundedArray)
    assert isinstance(observation_spec.items_placed_spec, specs.BoundedArray)
    assert isinstance(observation_spec.action_mask_spec, specs.BoundedArray)
    assert isinstance(binpack_env.action_spec(), specs.BoundedArray)


def test_binpack__step(binpack_env: BinPack) -> None:
    """Validates the jitted step of the environment."""
    # TODO: Do more checks when step function implemented.
    chex.clear_trace_counter()
    step_fn = jax.jit(chex.assert_max_traces(binpack_env.step, n=1))

    key = random.PRNGKey(0)
    state, timestep, _ = binpack_env.reset(key)

    action = binpack_env.action_spec().generate_value()
    state, timestep, _ = step_fn(state, action)
    # Call again to check it does not compile twice.
    state, timestep, _ = step_fn(state, action)


# TODO: Add below test once random binpack agent has been created.
# def test_binpack__does_not_smoke(
#     binpack_env: BinPack
# ) -> None:
#     """Test that we can run an episode without any errors."""
#     check_env_does_not_smoke(binpack_env)
