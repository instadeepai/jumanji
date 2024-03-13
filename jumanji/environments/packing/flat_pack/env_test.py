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

from jumanji.environments.packing.flat_pack.env import FlatPack
from jumanji.environments.packing.flat_pack.generator import (
    RandomFlatPackGenerator,
    ToyFlatPackGeneratorNoRotation,
    ToyFlatPackGeneratorWithRotation,
)
from jumanji.environments.packing.flat_pack.reward import (
    BlockDenseReward,
    CellDenseReward,
)
from jumanji.environments.packing.flat_pack.types import State
from jumanji.testing.env_not_smoke import check_env_does_not_smoke
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import StepType, TimeStep


@pytest.fixture(scope="module")
def flat_pack() -> FlatPack:
    """Creates a simple FlatPack environment for testing."""
    return FlatPack(
        generator=RandomFlatPackGenerator(
            num_col_blocks=3,
            num_row_blocks=3,
        ),
    )


@pytest.fixture
def simple_env_grid_state_1() -> chex.Array:
    """The state of the grid in the simplified example after 1 correct action."""
    # fmt: off
    return jnp.array(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    # fmt: on


@pytest.fixture
def simple_env_grid_state_2() -> chex.Array:
    """The state of the grid in the simplified example after 2 correct actions."""
    # fmt: off
    return jnp.array(
        [
            [1, 1, 1, 2, 2],
            [1, 1, 2, 2, 2],
            [0, 1, 0, 0, 2],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    # fmt: on


@pytest.fixture
def simple_env_grid_state_3() -> chex.Array:
    """The state of the grid in the simplified example after 3 correct actions."""
    # fmt: off
    return jnp.array(
        [
            [1, 1, 1, 2, 2],
            [1, 1, 2, 2, 2],
            [3, 1, 0, 0, 2],
            [3, 3, 0, 0, 0],
            [3, 3, 3, 0, 0],
        ]
    )
    # fmt: on


@pytest.fixture
def simple_env_grid_state_4() -> chex.Array:
    """The state of the grdi in the simplified example after 4 correct actions."""
    # fmt: off
    return jnp.array(
        [
            [1, 1, 1, 2, 2],
            [1, 1, 2, 2, 2],
            [3, 1, 4, 4, 2],
            [3, 3, 4, 4, 4],
            [3, 3, 3, 4, 4],
        ]
    )
    # fmt: on


@pytest.fixture
def simple_env_placed_blocks_1() -> chex.Array:
    """Placed blocks array in the simplified env after 1 block has been placed."""
    return jnp.array([True, False, False, False])


@pytest.fixture
def simple_env_placed_blocks_2() -> chex.Array:
    """Placed blocks array in the simplified env after 2 blocks have been placed."""
    return jnp.array([True, True, False, False])


@pytest.fixture
def simple_env_placed_blocks_3() -> chex.Array:
    """Placed blocks array in the simplified env after 3 blocks have been placed."""
    return jnp.array([True, True, True, False])


def test_flat_pack__reset_jit(flat_pack: FlatPack, key: chex.PRNGKey) -> None:
    """Test that the environment reset only compiles once."""
    chex.clear_trace_counter()
    reset_fn = jax.jit(chex.assert_max_traces(flat_pack.reset, n=1))
    state, timestep = reset_fn(key)

    # Check the types of the outputs
    assert isinstance(state, State)
    assert isinstance(timestep, TimeStep)

    # Check that the state contains DeviceArrays to verify that it is jitted.
    assert_is_jax_array_tree(state)

    # Call the reset method again to ensure it is not compiling twice.
    key, new_key = jax.random.split(key)
    state, timestep = reset_fn(new_key)
    assert isinstance(state, State)
    assert isinstance(timestep, TimeStep)


def test_flat_pack__step_jit(flat_pack: FlatPack, key: chex.PRNGKey) -> None:
    """Test that the step function is only compiled once."""
    state_0, timestep_0 = flat_pack.reset(key)
    action_0 = jnp.array([0, 0, 0, 0])

    chex.clear_trace_counter()
    step_fn = jax.jit(chex.assert_max_traces(flat_pack.step, n=1))

    state_1, timestep_1 = step_fn(state_0, action_0)

    # Check that the state has changed and that the step has incremented.
    assert not jnp.array_equal(state_1.grid, state_0.grid)
    assert state_1.step_count == state_0.step_count + 1
    assert isinstance(timestep_1, TimeStep)

    # Check that the state contains DeviceArrays to verify that it is jitted.
    assert_is_jax_array_tree(state_1)

    # Call the step method again to ensure it is not compiling twice.
    action_1 = jnp.array([1, 0, 3, 3])
    state_2, timestep_2 = step_fn(state_1, action_1)

    # Check that the state contains DeviceArrays to verify that it is jitted.
    assert_is_jax_array_tree(state_2)

    # Check that the state has changed and that the step has incremented.
    assert not jnp.array_equal(state_2.grid, state_1.grid)
    assert state_2.step_count == state_1.step_count + 1
    assert isinstance(timestep_2, TimeStep)


def test_flat_pack__does_not_smoke(flat_pack: FlatPack) -> None:
    """Test that we can run an episode without any errors."""
    check_env_does_not_smoke(flat_pack)


def test_flat_pack__is_done(flat_pack: FlatPack, key: chex.PRNGKey) -> None:
    """Test that the is_done method works as expected."""

    state, _ = flat_pack.reset(key)
    assert not flat_pack._is_done(state)

    # Manually set step count equal to the number of blocks.
    state.step_count = 9
    assert flat_pack._is_done(state)


def test_flat_pack__expand_block_to_grid(
    flat_pack: FlatPack, key: chex.PRNGKey, block: chex.Array
) -> None:
    """Test that a block is correctly set on a grid of zeros."""
    _, _ = flat_pack.reset(key)
    expanded_grid_with_block = flat_pack._expand_block_to_grid(block, 2, 1)
    # fmt: off
    expected_expanded_grid = jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    # fmt: on
    assert jnp.array_equal(expanded_grid_with_block, expected_expanded_grid)


def test_flat_pack__completed_episode_with_cell_dense_reward(
    key: chex.PRNGKey,
    simple_env_grid_state_1: chex.Array,
    simple_env_grid_state_2: chex.Array,
    simple_env_grid_state_3: chex.Array,
    simple_env_grid_state_4: chex.Array,
    simple_env_placed_blocks_1: chex.Array,
    simple_env_placed_blocks_2: chex.Array,
    simple_env_placed_blocks_3: chex.Array,
) -> None:
    """This test will step a simplified version of the FlatPack environment
    with a cell dense reward until completion. It will check that the reward is
    correctly computed and that the environment transitions as expected until
    done.
    """

    simple_env = FlatPack(
        generator=ToyFlatPackGeneratorNoRotation(),
        reward_fn=CellDenseReward(),
    )
    chex.clear_trace_counter()
    step_fn = jax.jit(chex.assert_max_traces(simple_env.step, n=1))

    # Intialize the environment
    state, timestep = simple_env.reset(key)
    assert isinstance(state, State)
    assert isinstance(timestep, TimeStep)
    assert timestep.step_type == StepType.FIRST

    # Check that the reset board contains only zeros
    assert jnp.all(state.grid == 0)
    assert jnp.all(state.action_mask)

    # Step the environment
    state, timestep = step_fn(state, jnp.array([0, 0, 0, 0]))
    assert timestep.step_type == StepType.MID
    assert jnp.all(state.grid == simple_env_grid_state_1)
    assert timestep.reward == 6 / 25
    assert jnp.all(state.placed_blocks == simple_env_placed_blocks_1)

    # Step the environment
    state, timestep = step_fn(state, jnp.array([1, 0, 0, 2]))
    assert timestep.step_type == StepType.MID
    assert jnp.all(state.grid == simple_env_grid_state_2)
    assert timestep.reward == 6 / 25
    assert jnp.all(state.placed_blocks == simple_env_placed_blocks_2)

    # Step the environment
    state, timestep = step_fn(state, jnp.array([2, 0, 2, 0]))
    assert timestep.step_type == StepType.MID
    assert jnp.all(state.grid == simple_env_grid_state_3)
    assert timestep.reward == 6 / 25
    assert jnp.all(state.placed_blocks == simple_env_placed_blocks_3)

    # Step the environment
    state, timestep = step_fn(state, jnp.array([3, 0, 2, 2]))
    assert timestep.step_type == StepType.LAST
    assert jnp.all(state.grid == simple_env_grid_state_4)
    assert timestep.reward == 7 / 25
    assert jnp.all(~state.action_mask)


def test_flat_pack__completed_episode_with_sparse_block_dense_reward(
    key: chex.PRNGKey,
    simple_env_grid_state_1: chex.Array,
    simple_env_grid_state_2: chex.Array,
    simple_env_grid_state_3: chex.Array,
    simple_env_grid_state_4: chex.Array,
    simple_env_placed_blocks_1: chex.Array,
    simple_env_placed_blocks_2: chex.Array,
    simple_env_placed_blocks_3: chex.Array,
) -> None:
    """This test will step a simplified version of the FlatPack environment
    with a block dense reward until completion. It will check that the reward is
    correctly computed and that the environment transitions as expected until
    done.
    """

    simple_env = FlatPack(
        generator=ToyFlatPackGeneratorWithRotation(),
        reward_fn=BlockDenseReward(),
    )
    chex.clear_trace_counter()
    step_fn = jax.jit(chex.assert_max_traces(simple_env.step, n=1))

    # Intialize the environment
    state, timestep = simple_env.reset(key)
    assert isinstance(state, State)
    assert isinstance(timestep, TimeStep)
    assert timestep.step_type == StepType.FIRST

    # Check that the reset board contains only zeros
    assert jnp.all(state.grid == 0)
    assert jnp.all(state.action_mask)

    # Step the environment
    state, timestep = step_fn(state, jnp.array([0, 2, 0, 0]))
    assert timestep.step_type == StepType.MID
    assert jnp.all(state.grid == simple_env_grid_state_1)
    assert timestep.reward == 1 / 4
    assert jnp.all(state.placed_blocks == simple_env_placed_blocks_1)

    # Step the environment
    state, timestep = step_fn(state, jnp.array([1, 2, 0, 2]))
    assert timestep.step_type == StepType.MID

    assert jnp.all(state.grid == simple_env_grid_state_2)
    assert timestep.reward == 1 / 4
    assert jnp.all(state.placed_blocks == simple_env_placed_blocks_2)

    # Step the environment
    state, timestep = step_fn(state, jnp.array([2, 1, 2, 0]))
    assert timestep.step_type == StepType.MID
    assert jnp.all(state.grid == simple_env_grid_state_3)
    assert timestep.reward == 1 / 4
    assert jnp.all(state.placed_blocks == simple_env_placed_blocks_3)

    # Step the environment
    state, timestep = step_fn(state, jnp.array([3, 0, 2, 2]))
    assert timestep.step_type == StepType.LAST
    assert jnp.all(state.grid == simple_env_grid_state_4)
    assert timestep.reward == 1 / 4
    assert jnp.all(~state.action_mask)
