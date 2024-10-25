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
from jumanji.environments.packing.bin_pack.env import BinPack, ExtendedBinPack
from jumanji.environments.packing.bin_pack.generator import (
    TWENTY_FOOT_DIMS,
    ExtendedRandomGenerator,
    ExtendedToyGenerator,
    Generator,
    RandomGenerator,
    ToyGenerator,
    make_container,
)
from jumanji.environments.packing.bin_pack.reward import (
    DenseReward,
    SparseReward,
    ValueBasedDenseReward,
    ValueBasedSparseReward,
)
from jumanji.environments.packing.bin_pack.space import Space
from jumanji.environments.packing.bin_pack.types import (
    Item,
    Location,
    State,
    ValuedItem,
)


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
            nb_items=3,
            items_mask=jnp.array([True, True, True], bool),
            items_placed=jnp.array([False, False, False], bool),
            items_location=jax.tree_util.tree_map(
                lambda x: jnp.array(3 * [x], jnp.int32), Location(x=0, y=0, z=0)
            ),
            action_mask=None,
            sorted_ems_indexes=jnp.arange(self.max_num_ems, dtype=jnp.int32),
            # For non value based optimisation set these to dummy values by default
            instance_max_item_value_magnitude=0.0,
            instance_total_value=0.0,
            # For deterministic instance generators we always set the key to 0.
            key=jax.random.PRNGKey(0),
        )


class DummyValueGenerator(Generator):
    """Dummy instance generator used for testing. It outputs a constant instance with a 20-ft
    container and 3 items: two identical items and a different third one to be able to
    test item aggregation.
    """

    def __init__(self) -> None:
        """Instantiate a dummy `Generator` with 3 items and 10 EMSs maximum."""
        super(DummyValueGenerator, self).__init__(
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
            items=ValuedItem(
                # The 1st and 2nd items have the same shape and value.
                x_len=jnp.array([1000, 1000, 500], jnp.int32),
                y_len=jnp.array([700, 700, 500], jnp.int32),
                z_len=jnp.array([900, 900, 600], jnp.int32),
                value=jnp.array([2.0, 2.0, 1.5], jnp.float32),
            ),
            items_mask=jnp.array([True, True, True], bool),
            items_placed=jnp.array([False, False, False], bool),
            items_location=jax.tree_util.tree_map(
                lambda x: jnp.array(3 * [x], jnp.int32), Location(x=0, y=0, z=0)
            ),
            action_mask=None,
            sorted_ems_indexes=jnp.arange(self.max_num_ems, dtype=jnp.int32),
            # For non value based optimisation set these to dummy values by default
            instance_max_item_value_magnitude=2.0,
            instance_total_value=5.5,
            # For deterministic instance generators we always set the key to 0.
            key=jax.random.PRNGKey(0),
            nb_items=3,
        )


class DummyExtendedGenerator(DummyGenerator):
    """Dummy instance generator used for testing. It outputs a constant instance with a 20-ft
    container and 3 items that can take all 6 possible orientations: two identical items and a
    different third one.
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
            items=ValuedItem(
                # The 1st and 2nd items have the same shape.
                x_len=jnp.array(
                    [
                        [1000, 1000, 500],
                        [1000, 1000, 500],
                        [700, 700, 500],
                        [700, 700, 500],
                        [900, 900, 600],
                        [900, 900, 600],
                    ],
                    jnp.int32,
                ),
                y_len=jnp.array(
                    [
                        [700, 700, 500],
                        [900, 900, 600],
                        [1000, 1000, 500],
                        [900, 900, 600],
                        [700, 700, 500],
                        [1000, 1000, 500],
                    ],
                    jnp.int32,
                ),
                z_len=jnp.array(
                    [
                        [900, 900, 600],
                        [700, 700, 500],
                        [900, 900, 600],
                        [1000, 1000, 500],
                        [1000, 1000, 500],
                        [700, 700, 500],
                    ],
                    jnp.int32,
                ),
                value=jnp.array(
                    [
                        [2.0, 2.0, 1.5],
                        [2.0, 2.0, 1.5],
                        [2.0, 2.0, 1.5],
                        [2.0, 2.0, 1.5],
                        [2.0, 2.0, 1.5],
                        [2.0, 2.0, 1.5],
                    ],
                    jnp.float32,
                ),
            ),
            items_mask=jnp.array(
                [
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                ],
                bool,
            ),
            items_placed=jnp.array(
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
                bool,
            ),
            items_location=jax.tree_util.tree_map(
                lambda x: jnp.array(3 * [x], jnp.int32), Location(x=0, y=0, z=0)
            ),
            instance_max_item_value_magnitude=2.0,
            instance_total_value=5.5,
            action_mask=None,
            sorted_ems_indexes=jnp.arange(self.max_num_ems, dtype=jnp.int32),
            # For deterministic instance generators we always set the key to 0.
            key=jax.random.PRNGKey(0),
            nb_items=3,
        )


class DummyRotationGenerator(DummyGenerator):
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
                x_len=jnp.array(
                    [
                        [1000, 1000, 500],
                        [1000, 1000, 500],
                        [700, 700, 500],
                        [700, 700, 500],
                        [900, 900, 600],
                        [900, 900, 600],
                    ],
                    jnp.int32,
                ),
                y_len=jnp.array(
                    [
                        [700, 700, 500],
                        [900, 900, 600],
                        [1000, 1000, 500],
                        [900, 900, 600],
                        [700, 700, 500],
                        [1000, 1000, 500],
                    ],
                    jnp.int32,
                ),
                z_len=jnp.array(
                    [
                        [900, 900, 600],
                        [700, 700, 500],
                        [900, 900, 600],
                        [1000, 1000, 500],
                        [1000, 1000, 500],
                        [700, 700, 500],
                    ],
                    jnp.int32,
                ),
            ),
            items_mask=jnp.array(
                [
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                ],
                bool,
            ),
            items_placed=jnp.array(
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
                bool,
            ),
            items_location=jax.tree_util.tree_map(
                lambda x: jnp.array(3 * [x], jnp.int32), Location(x=0, y=0, z=0)
            ),
            instance_max_item_value_magnitude=0,
            instance_total_value=0,
            action_mask=None,
            sorted_ems_indexes=jnp.arange(self.max_num_ems, dtype=jnp.int32),
            # For deterministic instance generators we always set the key to 0.
            key=jax.random.PRNGKey(0),
            nb_items=3,
        )


class FullSupportDummyGenerator(Generator):
    """Dummy instance generator used for testing. It outputs a constant instance with a 20-ft
    container and 11 items: 10 identical small items of size
    (container_length/10, container_width, 300) and one big item of size
    (container_length, container_width, 1900).
    This instance is used to test the full support constraint by forcing the agent to start by
    placing one of the small items. Using this instance allows us to test both that the agent isn't
    able to place items if they're not fully supported, and make sure that
    the merger of ems is correct.
    """

    def __init__(self) -> None:
        """Instantiate a dummy `Generator` with 3 items and 10 EMSs maximum."""
        super(FullSupportDummyGenerator, self).__init__(
            max_num_items=11, max_num_ems=40, container_dims=TWENTY_FOOT_DIMS
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
            instance_max_item_value_magnitude=0,
            instance_total_value=0,
            ems=jax.tree_util.tree_map(
                lambda x: jnp.array([x] + (self.max_num_ems - 1) * [0], jnp.int32),
                container,
            ),
            ems_mask=jnp.array([True] + (self.max_num_ems - 1) * [False], bool),
            items=Item(
                x_len=jnp.array([container.x2] + 10 * [container.x2 / 10], jnp.int32),
                y_len=jnp.array(11 * [container.y2], jnp.int32),
                z_len=jnp.array([1900] + 10 * [300], jnp.int32),
            ),
            items_mask=jnp.array(
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
                bool,
            ),
            items_placed=jnp.array(
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                bool,
            ),
            items_location=jax.tree_util.tree_map(
                lambda loc: jnp.array(11 * [loc], jnp.int32), Location(x=0, y=0, z=0)
            ),
            nb_items=11,
            action_mask=None,
            sorted_ems_indexes=jnp.arange(self.max_num_ems, dtype=jnp.int32),
            # For deterministic instance generators we always set the key to 0.
            key=jax.random.PRNGKey(0),
        )


@pytest.fixture
def dummy_generator() -> DummyGenerator:
    return DummyGenerator()


@pytest.fixture
def dummy_rotation_generator() -> DummyRotationGenerator:
    return DummyRotationGenerator()


@pytest.fixture
def dummy_extended_generator() -> DummyExtendedGenerator:
    return DummyExtendedGenerator()


@pytest.fixture
def toy_generator() -> ToyGenerator:
    return ToyGenerator()


@pytest.fixture
def random_generator() -> RandomGenerator:
    """Returns a `RandomGenerator` with up to 20 items and that can handle 80 EMSs."""
    return RandomGenerator(max_num_items=20, max_num_ems=80)


@pytest.fixture
def rotation_toy_generator() -> ExtendedToyGenerator:
    return ExtendedToyGenerator()


@pytest.fixture
def rotation_random_generator() -> ExtendedRandomGenerator:
    """Returns a `RandomGenerator` with up to 20 items and that can handle 80 EMSs."""
    return ExtendedRandomGenerator(
        max_num_items=20, max_num_ems=80, is_rotation_allowed=True, is_value_based=False
    )


@pytest.fixture
def dummy_state(dummy_generator: DummyGenerator) -> State:
    state = dummy_generator(key=jax.random.PRNGKey(0))
    num_ems = dummy_generator.max_num_ems
    num_items = dummy_generator.max_num_items
    state.action_mask = jnp.ones((num_ems, num_items), bool)
    return state


@pytest.fixture
def dummy_rotation_state(
    dummy_rotation_generator: DummyRotationGenerator,
) -> State:
    state = dummy_rotation_generator(key=jax.random.PRNGKey(0))
    num_ems = dummy_rotation_generator.max_num_ems
    num_items = dummy_rotation_generator.max_num_items
    state.action_mask = jnp.ones((6, num_ems, num_items), bool)
    return state


@pytest.fixture
def bin_pack(dummy_generator: DummyGenerator) -> BinPack:
    return BinPack(generator=dummy_generator, obs_num_ems=5)


@pytest.fixture()
def rotation_bin_pack(
    dummy_rotation_generator: DummyRotationGenerator,
) -> ExtendedBinPack:
    """
    Bin pack environment where the items can be rotated.
    """
    return ExtendedBinPack(
        generator=dummy_rotation_generator,
        obs_num_ems=5,
        is_rotation_allowed=True,
        is_value_based=False,
    )


@pytest.fixture()
def extended_bin_pack(
    dummy_extended_generator: DummyExtendedGenerator,
) -> ExtendedBinPack:
    """
    Bin pack environment where the items have a value and can be rotated.
    """
    return ExtendedBinPack(
        generator=dummy_extended_generator,
        obs_num_ems=5,
        is_rotation_allowed=True,
        is_value_based=True,
    )


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
def bin_pack_dense_value_reward() -> BinPack:
    return ExtendedBinPack(
        generator=DummyValueGenerator(),
        obs_num_ems=5,
        reward_fn=ValueBasedDenseReward(),
        is_value_based=True,
        is_rotation_allowed=False,
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


@pytest.fixture
def bin_pack_sparse_value_reward() -> BinPack:
    return ExtendedBinPack(
        generator=DummyValueGenerator(),
        obs_num_ems=5,
        reward_fn=ValueBasedSparseReward(),
        is_rotation_allowed=False,
        is_value_based=True,
    )


@pytest.fixture
def full_support_dummy_generator() -> FullSupportDummyGenerator:
    return FullSupportDummyGenerator()


@pytest.fixture
def full_support_bin_pack(
    full_support_dummy_generator: FullSupportDummyGenerator,
) -> BinPack:
    return BinPack(
        generator=full_support_dummy_generator, full_support=True, debug=True
    )
