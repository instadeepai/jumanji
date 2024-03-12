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

from jumanji.environments.packing.flat_pack.reward import (
    BlockDenseReward,
    CellDenseReward,
)
from jumanji.environments.packing.flat_pack.types import State


@pytest.fixture
def blocks() -> chex.Array:
    """An array containing 4 blocks."""

    return jnp.array(
        [
            [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 2.0, 2.0], [2.0, 2.0, 2.0], [0.0, 0.0, 2.0]],
            [[3.0, 0.0, 0.0], [3.0, 3.0, 0.0], [3.0, 3.0, 3.0]],
            [[4.0, 4.0, 0.0], [4.0, 4.0, 4.0], [0.0, 4.0, 4.0]],
        ],
        dtype=jnp.float32,
    )


@pytest.fixture()
def state_with_no_blocks_placed(
    solved_grid: chex.Array, key: chex.PRNGKey, blocks: chex.Array
) -> State:
    """A grid state with no blocks placed."""

    return State(
        num_blocks=4,
        blocks=blocks,
        action_mask=jnp.ones((4, 4, 2, 2), dtype=bool),
        placed_blocks=jnp.zeros(4, dtype=bool),
        grid=jnp.zeros_like(solved_grid),
        step_count=0,
        key=key,
    )


@pytest.fixture()
def state_with_block_one_placed(
    action_mask_with_block_1_placed: chex.Array,
    grid_with_block_one_placed: chex.Array,
    blocks: chex.Array,
    key: chex.PRNGKey,
) -> State:
    """A grid state with block one placed."""

    key, new_key = jax.random.split(key)
    return State(
        num_blocks=4,
        action_mask=action_mask_with_block_1_placed,
        placed_blocks=jnp.array(
            [
                True,
                False,
                False,
                False,
            ]
        ),
        grid=grid_with_block_one_placed,
        step_count=0,
        key=new_key,
        blocks=blocks,
    )


@pytest.fixture()
def state_needing_only_block_one(
    action_mask_without_only_block_1_placed: chex.Array,
    solved_grid: chex.Array,
    grid_with_block_one_placed: chex.Array,
    blocks: chex.Array,
    key: chex.PRNGKey,
) -> State:
    """A grid state that one needs block one to be fully completed."""

    key, new_key = jax.random.split(key)

    grid = solved_grid - grid_with_block_one_placed

    return State(
        num_blocks=4,
        action_mask=action_mask_without_only_block_1_placed,
        placed_blocks=jnp.array(
            [
                True,
                False,
                False,
                False,
            ]
        ),
        grid=grid,
        step_count=3,
        blocks=blocks,
        key=new_key,
    )


@pytest.fixture()
def solved_state(
    solved_grid: chex.Array,
    blocks: chex.Array,
    key: chex.PRNGKey,
) -> State:
    """A solved grid state."""

    key, new_key = jax.random.split(key)

    return State(
        num_blocks=4,
        action_mask=jnp.zeros((4, 4, 2, 2), dtype=bool),
        placed_blocks=jnp.array(
            [
                True,
                True,
                True,
                True,
            ]
        ),
        grid=solved_grid,
        step_count=4,
        blocks=blocks,
        key=new_key,
    )


@pytest.fixture()
def block_one_placed_at_2_2(grid_with_block_one_placed: chex.Array) -> chex.Array:
    """A 2D array of zeros where block one has been placed with it left top-most
    corner at position (2, 2).
    """

    # Shift all elements in the array two down and two to the right
    placed_block = jnp.roll(grid_with_block_one_placed, shift=2, axis=0)
    placed_block = jnp.roll(placed_block, shift=2, axis=1)

    return placed_block


def test_cell_dense_reward(
    state_with_no_blocks_placed: State,
    state_with_block_one_placed: State,
    block_one_placed_at_0_0: chex.Array,
    block_one_placed_at_1_1: chex.Array,
    block_one_placed_at_2_2: chex.Array,
) -> None:

    dense_reward = jax.jit(CellDenseReward())

    # Test placing block one completely correctly
    reward = dense_reward(
        state=state_with_no_blocks_placed,
        action=block_one_placed_at_0_0,
        is_valid=True,
        is_done=False,
        next_state=state_with_block_one_placed,
    )
    assert reward == 6.0 / 25.0

    # Test placing block one partially correct
    reward = dense_reward(
        state=state_with_no_blocks_placed,
        action=block_one_placed_at_1_1,
        is_valid=True,
        is_done=False,
        next_state=state_with_block_one_placed,
    )
    assert reward == 6.0 / 25.0

    # Test placing a completely incorrect block
    reward = dense_reward(
        state=state_with_no_blocks_placed,
        action=block_one_placed_at_2_2,
        is_valid=True,
        is_done=False,
        next_state=state_with_block_one_placed,
    )
    assert reward == 6.0 / 25.0

    # Test invalid action returns 0 reward.
    reward = dense_reward(
        state=state_with_no_blocks_placed,
        action=block_one_placed_at_0_0,
        is_valid=False,
        is_done=False,
        next_state=state_with_block_one_placed,
    )
    assert reward == 0.0


def test_block_dense_reward(
    state_with_no_blocks_placed: State,
    state_with_block_one_placed: State,
    block_one_placed_at_0_0: chex.Array,
    block_one_placed_at_1_1: chex.Array,
    block_one_placed_at_2_2: chex.Array,
) -> None:

    dense_reward = jax.jit(BlockDenseReward())

    # Test placing block one completely correctly
    reward = dense_reward(
        state=state_with_no_blocks_placed,
        action=block_one_placed_at_0_0,
        is_valid=True,
        is_done=False,
        next_state=state_with_block_one_placed,
    )
    assert reward == 1.0 / 4.0

    # Test placing block one partially correct
    reward = dense_reward(
        state=state_with_no_blocks_placed,
        action=block_one_placed_at_1_1,
        is_valid=True,
        is_done=False,
        next_state=state_with_block_one_placed,
    )
    assert reward == 1.0 / 4.0

    # Test placing a completely incorrect block
    reward = dense_reward(
        state=state_with_no_blocks_placed,
        action=block_one_placed_at_2_2,
        is_valid=True,
        is_done=False,
        next_state=state_with_block_one_placed,
    )
    assert reward == 1.0 / 4.0

    # Test invalid action returns 0 reward.
    reward = dense_reward(
        state=state_with_no_blocks_placed,
        action=block_one_placed_at_0_0,
        is_valid=False,
        is_done=False,
        next_state=state_with_block_one_placed,
    )
    assert reward == 0.0
