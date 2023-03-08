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

import jax
import jax.numpy as jnp

import jumanji.tree_utils
from jumanji.environments.packing.bin_pack.env import BinPack
from jumanji.environments.packing.bin_pack.reward import DenseReward, SparseReward
from jumanji.environments.packing.bin_pack.types import item_volume


def test__sparse_reward(
    bin_pack_sparse_reward: BinPack, sparse_reward: SparseReward
) -> None:
    reward_fn = jax.jit(sparse_reward)
    step_fn = jax.jit(bin_pack_sparse_reward.step)
    state, timestep = bin_pack_sparse_reward.reset(jax.random.PRNGKey(0))

    # Check that the reward is correct for the next item.
    for item_id, is_valid in enumerate(timestep.observation.items_mask):
        action = jnp.array([0, item_id], jnp.int32)
        next_state, next_timestep = step_fn(state, action)
        reward = reward_fn(
            state, action, next_state, is_valid, is_done=next_timestep.last()
        )
        assert reward == next_timestep.reward == 0

    # Check that all other invalid actions lead to the 0 reward, any ems_id > 0 is not valid at
    # the beginning of the episode.
    for ems_id in range(1, timestep.observation.action_mask.shape[0]):
        for item_id in range(timestep.observation.action_mask.shape[1]):
            action = jnp.array([ems_id, item_id], jnp.int32)
            next_state, next_timestep = step_fn(state, action)
            is_valid = timestep.observation.action_mask[tuple(action)]
            is_done = next_timestep.last()
            assert ~is_valid and is_done
            reward = reward_fn(state, action, next_state, is_valid, is_done)
            assert reward == 0 == next_timestep.reward

    # Check that taking an invalid action after packing one item returns the utilization
    # of the first item.
    action = jnp.array([0, 0], jnp.int32)
    state, timestep = step_fn(state, action)
    assert timestep.reward == 0
    assert timestep.mid()
    next_state, timestep = step_fn(state, action)
    reward = reward_fn(state, action, next_state, is_valid=False, is_done=True)
    assert timestep.last()
    item = jumanji.tree_utils.tree_slice(timestep.observation.items, action[1])
    assert jnp.isclose(reward, item_volume(item))


def test_dense_reward(
    bin_pack_dense_reward: BinPack, dense_reward: DenseReward
) -> None:
    reward_fn = jax.jit(dense_reward)
    step_fn = jax.jit(bin_pack_dense_reward.step)
    state, timestep = bin_pack_dense_reward.reset(jax.random.PRNGKey(0))

    # Check that the reward is correct for the next item.
    for item_id, is_valid in enumerate(timestep.observation.items_mask):
        action = jnp.array([0, item_id], jnp.int32)
        next_state, next_timestep = step_fn(state, action)
        reward = reward_fn(
            state, action, next_state, is_valid, is_done=next_timestep.last()
        )
        assert reward == next_timestep.reward
        if is_valid:
            item = jumanji.tree_utils.tree_slice(timestep.observation.items, item_id)
            assert jnp.isclose(reward, item_volume(item))
        else:
            assert reward == 0
            assert next_timestep.last()

    # Check that all other invalid actions lead to the 0 reward.
    for ems_id in range(1, timestep.observation.action_mask.shape[0]):
        for item_id in range(timestep.observation.action_mask.shape[1]):
            action = jnp.array([ems_id, item_id], jnp.int32)
            next_state, next_timestep = step_fn(state, action)
            is_valid = timestep.observation.action_mask[tuple(action)]
            is_done = next_timestep.last()
            assert ~is_valid and is_done
            reward = reward_fn(state, action, next_state, is_valid, is_done)
            assert reward == 0 == next_timestep.reward
