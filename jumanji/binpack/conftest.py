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

import functools

import chex
import jax
import jax.numpy as jnp
import pytest

from jumanji.binpack.env import BinPack
from jumanji.binpack.generator import (
    TWENTY_FOOT_DIMS,
    InstanceGenerator,
    normalized_container,
)
from jumanji.binpack.specs import ObservationSpec
from jumanji.binpack.types import Location, State, item_from_space


class DummyInstanceGenerator(InstanceGenerator):
    """Dummy instance generator used for testing. It outputs a constant instance with a 20-ft
    container and one item whose size is the container itself.
    """

    def __init__(self) -> None:
        """Instantiate a dummy `InstanceGenerator` with one item and one EMS maximum."""
        super(DummyInstanceGenerator, self).__init__(max_num_items=1, max_num_ems=1)
        self.container_dims = TWENTY_FOOT_DIMS

    def __call__(self, key: chex.PRNGKey) -> State:
        """Returns a fixed instance with one item, one EMS and a 20-ft container.

        Args:
            key: random key not used here but kept for consistency with parent signature.

        Returns:
            State.
        """
        del key
        container = normalized_container(self.container_dims)
        return State(
            container=container,
            ems=jax.tree_map(
                functools.partial(jnp.expand_dims, axis=-1),
                container,
            ),
            ems_mask=jnp.array([True], bool),
            # TODO: batch items
            items=jax.tree_map(
                functools.partial(jnp.expand_dims, axis=-1),
                item_from_space(container),
            ),
            items_mask=jnp.array([True], bool),
            items_placed=jnp.array([False], bool),
            items_location=jax.tree_map(
                lambda x: jnp.array([x], dtype=jnp.float32), Location(x=0, y=0, z=0)
            ),
            action_mask=None,
            sorted_ems_indexes=jnp.array([0], int),
        )


@pytest.fixture
def dummy_instance_generator() -> DummyInstanceGenerator:
    return DummyInstanceGenerator()


@pytest.fixture
def dummy_instance(dummy_instance_generator: DummyInstanceGenerator) -> State:
    return dummy_instance_generator(key=jax.random.PRNGKey(0))


@pytest.fixture
def binpack_env(dummy_instance_generator: DummyInstanceGenerator) -> BinPack:
    return BinPack(instance_generator=dummy_instance_generator, obs_num_ems=1)


@pytest.fixture
def obs_spec(binpack_env: BinPack) -> ObservationSpec:
    return binpack_env.observation_spec()
