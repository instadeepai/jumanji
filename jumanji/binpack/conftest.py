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

import jax
import jax.numpy as jnp
import pytest
from chex import PRNGKey

from jumanji.binpack.env import BinPack
from jumanji.binpack.generator import Generator
from jumanji.binpack.specs import ObservationSpec
from jumanji.binpack.types import EMS, Container, Item, Location, State


class DummyGenerator(Generator):
    """Dummy generator used for testing. It outputs a constant instance with a cubic container and
    one item whose size is the container itself.
    """

    def __init__(self) -> None:
        """Instantiate a dummy generator with one item and one ems maximum."""
        super(DummyGenerator, self).__init__(max_num_items=1, max_num_ems=1)

    def __call__(self, key: PRNGKey) -> State:
        """Returns a fixed instance with one item, one ems and a cubic container.

        Args:
            key: random key not used here but kept for consistency with parent signature.

        Returns:
            State.
        """
        del key
        return State(
            container=Container(x1=0, x2=1, y1=0, y2=1, z1=0, z2=1).astype(float),
            ems=jax.tree_map(
                functools.partial(jnp.expand_dims, axis=-1),
                EMS(x1=0, x2=1, y1=0, y2=1, z1=0, z2=1).astype(float),
            ),
            ems_mask=jnp.array([True], bool),
            items=jax.tree_map(
                functools.partial(jnp.asarray, dtype=float),
                Item(x_len=1, y_len=1, z_len=1),
            ),
            items_mask=jnp.array([True], bool),
            items_placed=jnp.array([False], bool),
            items_location=jax.tree_map(
                functools.partial(jnp.asarray, dtype=float), Location(x=0, y=0, z=0)
            ),
            action_mask=jnp.array([[True]], bool),
            sorted_ems_indexes=jnp.array([0], int),
        )


@pytest.fixture
def dummy_generator() -> DummyGenerator:
    return DummyGenerator()


@pytest.fixture
def binpack_env(dummy_generator: Generator) -> BinPack:
    return BinPack(generator=dummy_generator, obs_num_ems=1)


@pytest.fixture
def obs_spec(binpack_env: BinPack) -> ObservationSpec:
    return binpack_env.observation_spec()
