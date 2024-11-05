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

from jumanji.environments.packing.knapsack.env import Knapsack
from jumanji.environments.packing.knapsack.reward import DenseReward, SparseReward


def test_dense_reward(knapsack_dense_reward: Knapsack, dense_reward: DenseReward) -> None:
    dense_reward = jax.jit(dense_reward)
    step_fn = jax.jit(knapsack_dense_reward.step)
    state, timestep = knapsack_dense_reward.reset(jax.random.PRNGKey(0))

    # Check that the reward is correct for any item.
    for action in range(knapsack_dense_reward.num_items):
        next_state, _ = step_fn(state, action)
        item_value = state.values[action]
        reward = dense_reward(state, action, next_state, is_valid=True, is_done=False)
        assert reward == item_value

    # Check the reward for invalid action.
    next_state, _ = step_fn(state, 0)
    reward = dense_reward(state, 0, next_state, is_valid=False, is_done=True)
    assert reward == 0


def test_sparse_reward(knapsack_sparse_reward: Knapsack, sparse_reward: SparseReward) -> None:
    sparse_reward = jax.jit(sparse_reward)
    step_fn = jax.jit(knapsack_sparse_reward.step)
    state, timestep = knapsack_sparse_reward.reset(jax.random.PRNGKey(0))

    # Check that all but the last step leads to 0 reward.
    next_state = state
    while not timestep.last():
        for action, is_valid in enumerate(timestep.observation.action_mask):
            if is_valid:
                next_state, timestep = step_fn(state, action)
                reward = sparse_reward(state, action, next_state, is_valid, is_done=timestep.last())
                if timestep.last():
                    # At the end of the episode, check that the reward is the total values of
                    # packed items.
                    total_value = jnp.sum(state.values, where=next_state.packed_items)
                    assert reward == total_value
                else:
                    # Check that the reward is 0 for every non-final valid action.
                    assert reward == 0
            else:
                # Check that the reward is also 0 for every invalid action.
                invalid_next_state, _ = step_fn(state, action)
                reward = sparse_reward(
                    state, action, invalid_next_state, is_valid, is_done=timestep.last()
                )
                assert reward == 0
        state = next_state
