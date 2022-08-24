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

from typing import Type

import chex
import jax
import jax.numpy as jnp
import pytest
from jax import random

from jumanji import specs
from jumanji.environments.combinatorial.binpack.env import BinPack
from jumanji.environments.combinatorial.binpack.instance_generator import (
    InstanceGenerator,
)
from jumanji.environments.combinatorial.binpack.specs import (
    EMSSpec,
    ItemSpec,
    ObservationSpec,
)
from jumanji.environments.combinatorial.binpack.types import Observation, State
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import TimeStep


def assert_type_binpack_state(state: State) -> None:
    """Assert that all spaces or items are integers while all masks are boolean in the state."""
    jax.tree_map(
        lambda leaf: chex.assert_type(leaf, int),
        (
            state.container,
            state.ems,
            state.items,
            state.items_location,
            state.sorted_ems_indexes,
        ),
    )
    jax.tree_map(
        lambda leaf: chex.assert_type(leaf, bool),
        (state.ems_mask, state.items_mask, state.items_placed, state.action_mask),
    )


def test_binpack__reset(binpack_env: BinPack) -> None:
    """Validates the jitted reset of the environment."""
    chex.clear_trace_counter()
    reset_fn = jax.jit(chex.assert_max_traces(binpack_env.reset, n=1))

    key = random.PRNGKey(0)
    state, timestep = reset_fn(key)
    # Call again to check it does not compile twice.
    state, timestep = reset_fn(key)
    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # reset function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(state)
    assert_type_binpack_state(state)
    assert state.ems_mask.any()
    assert state.items_mask.any()
    assert jnp.any(state.action_mask)


@pytest.mark.parametrize("normalize_dimensions", [True, False])
def test_binpack__spec(
    normalize_dimensions: bool, dummy_instance_generator: InstanceGenerator
) -> None:
    """Validates the observation and action spec of the BinPack environment. Checks that
    different specs are generated depending on the `normalize_dimensions` argument.
    """
    binpack_env = BinPack(
        instance_generator=dummy_instance_generator,
        obs_num_ems=1,
        normalize_dimensions=normalize_dimensions,
    )
    observation_spec = binpack_env.observation_spec()
    assert isinstance(observation_spec, ObservationSpec)
    assert isinstance(observation_spec.ems_spec, EMSSpec)
    assert isinstance(observation_spec.ems_mask_spec, specs.BoundedArray)
    assert isinstance(observation_spec.items_spec, ItemSpec)
    assert isinstance(observation_spec.items_mask_spec, specs.BoundedArray)
    assert isinstance(observation_spec.items_placed_spec, specs.BoundedArray)
    assert isinstance(observation_spec.action_mask_spec, specs.BoundedArray)
    assert isinstance(binpack_env.action_spec(), specs.MultiDiscreteArray)
    observation = observation_spec.generate_value()

    def assert_type_binpack_observation(obs: Observation, type_: Type) -> None:
        """Assert that the EMS and items are of given type."""
        jax.tree_map(lambda leaf: chex.assert_type(leaf, type_), (obs.ems, obs.items))

    if normalize_dimensions:
        assert_type_binpack_observation(observation, float)
    else:
        assert_type_binpack_observation(observation, int)


def test_binpack__step(binpack_env: BinPack) -> None:
    """Validates the jitted step of the environment."""
    # TODO: Do more checks when step function implemented.
    chex.clear_trace_counter()
    step_fn = jax.jit(chex.assert_max_traces(binpack_env.step, n=1))

    key = random.PRNGKey(0)
    state, timestep = binpack_env.reset(key)

    action = binpack_env.action_spec().generate_value()
    state, timestep = step_fn(state, action)
    # Call again to check it does not compile twice.
    state, timestep = step_fn(state, action)
    assert_type_binpack_state(state)


def test_binpack__render_does_not_smoke(
    binpack_env: BinPack, dummy_instance: State
) -> None:
    binpack_env.render(dummy_instance)
    binpack_env.close()


# TODO: Add below test once random binpack agent has been created.
# def test_binpack__does_not_smoke(
#     binpack_env: BinPack
# ) -> None:
#     """Test that we can run an episode without any errors."""
#     check_env_does_not_smoke(binpack_env)
