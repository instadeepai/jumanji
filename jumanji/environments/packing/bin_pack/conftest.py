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

from jumanji import specs
from jumanji.environments.packing.bin_pack.env import BinPack
from jumanji.environments.packing.bin_pack.generator import (
    TWENTY_FOOT_DIMS,
    Generator,
    RandomGenerator,
    ToyGenerator,
    make_container,
)
from jumanji.environments.packing.bin_pack.reward import DenseReward, SparseReward
from jumanji.environments.packing.bin_pack.space import Space
from jumanji.environments.packing.bin_pack.types import Item, Location, State


class DummyGenerator(Generator):
    """Dummy instance generator used for testing. It outputs a constant instance with a 20-ft
    container and 3 items: two identical items and a different third one to be able to
    test item aggregation.
    """

    def __init__(self) -> None:
        """Instantiate a dummy `Generator` with 3 items and 10 EMSs maximum."""
        super(DummyGenerator, self).__init__(
            max_num_items=3, max_num_ems=10, container_dims=TWENTY_FOOT_DIMS
        )

    def __call__(self, key: chex.PRNGKey) -> State:
        """Returns a fixed instance with 3 items, 10 EMSs and a 20-ft container.

        Args:
            key: random key not used here but kept for consistency with parent signature.

        Returns:
            State.
        """
        del key
        container = make_container(TWENTY_FOOT_DIMS)
        return State(
            container=container,
            ems=jax.tree_util.tree_map(
                lambda x: jnp.array([x] + (self.max_num_ems - 1) * [0], jnp.int32),
                container,
            ),
            ems_mask=jnp.array([True] + (self.max_num_ems - 1) * [False], bool),
            items=Item(
                # The 1st and 2nd items have the same shape.
                x_len=jnp.array([1000, 1000, 500], jnp.int32),
                y_len=jnp.array([700, 700, 500], jnp.int32),
                z_len=jnp.array([900, 900, 600], jnp.int32),
            ),
            items_mask=jnp.array([True, True, True], bool),
            items_placed=jnp.array([False, False, False], bool),
            items_location=jax.tree_util.tree_map(
                lambda x: jnp.array(3 * [x], jnp.int32), Location(x=0, y=0, z=0)
            ),
            action_mask=None,
            sorted_ems_indexes=jnp.arange(self.max_num_ems, dtype=jnp.int32),
            # For deterministic instance generators we always set the key to 0.
            key=jax.random.PRNGKey(0),
        )


@pytest.fixture
def dummy_generator() -> DummyGenerator:
    return DummyGenerator()


@pytest.fixture
def toy_generator() -> ToyGenerator:
    return ToyGenerator()


@pytest.fixture
def random_generator() -> RandomGenerator:
    """Returns a `RandomGenerator` with up to 20 items and that can handle 80 EMSs."""
    return RandomGenerator(max_num_items=20, max_num_ems=80)


@pytest.fixture
def dummy_state(dummy_generator: DummyGenerator) -> State:
    state = dummy_generator(key=jax.random.PRNGKey(0))
    num_ems = dummy_generator.max_num_ems
    num_items = dummy_generator.max_num_items
    state.action_mask = jnp.ones((num_ems, num_items), bool)
    return state


@pytest.fixture
def bin_pack(dummy_generator: DummyGenerator) -> BinPack:
    return BinPack(generator=dummy_generator, obs_num_ems=5)


@pytest.fixture
def obs_spec(bin_pack: BinPack) -> specs.Spec:
    return bin_pack.observation_spec


@pytest.fixture
def space() -> Space:
    return Space(x1=0, x2=1, y1=0, y2=1, z1=0, z2=1).astype(jnp.int32)


@pytest.fixture
def dense_reward() -> DenseReward:
    return DenseReward()


@pytest.fixture
def bin_pack_dense_reward(
    dummy_generator: DummyGenerator, dense_reward: DenseReward
) -> BinPack:
    return BinPack(
        generator=dummy_generator,
        obs_num_ems=5,
        reward_fn=dense_reward,
    )


@pytest.fixture
def sparse_reward() -> SparseReward:
    return SparseReward()


@pytest.fixture
def bin_pack_sparse_reward(
    dummy_generator: DummyGenerator, sparse_reward: SparseReward
) -> BinPack:
    return BinPack(
        generator=dummy_generator,
        obs_num_ems=5,
        reward_fn=sparse_reward,
    )
