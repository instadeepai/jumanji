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

from jumanji.environments.routing.tsp.env import TSP
from jumanji.environments.routing.tsp.reward import DenseReward, SparseReward


def test_dense_reward(tsp_dense_reward: TSP, dense_reward: DenseReward) -> None:
    dense_reward = jax.jit(dense_reward)
    step_fn = jax.jit(tsp_dense_reward.step)
    state, timestep = tsp_dense_reward.reset(jax.random.PRNGKey(0))

    # Check that the first action leads to 0 reward
    for action in range(tsp_dense_reward.num_cities):
        next_state, _ = step_fn(state, action)
        reward = dense_reward(state, action, next_state, is_valid=True)
        assert reward == 0

    # Check that the reward is correct for the next city
    state, timestep = step_fn(state, 0)
    for action in range(1, tsp_dense_reward.num_cities):
        next_state, _ = step_fn(state, action)
        initial_city = state.coordinates[0]
        new_city = state.coordinates[action]
        distance = jnp.linalg.norm(new_city - initial_city)
        reward = dense_reward(state, action, next_state, is_valid=True)
        assert reward == -distance

    # Check the reward for invalid action
    next_state, _ = step_fn(state, 0)
    penalty = -jnp.sqrt(2) * tsp_dense_reward.num_cities
    reward = dense_reward(state, 0, next_state, is_valid=False)
    assert reward == penalty


def test_sparse_reward(  # noqa: CCR001
    tsp_sparse_reward: TSP, sparse_reward: SparseReward
) -> None:
    sparse_reward = jax.jit(sparse_reward)
    step_fn = jax.jit(tsp_sparse_reward.step)
    state, timestep = tsp_sparse_reward.reset(jax.random.PRNGKey(0))
    penalty = -jnp.sqrt(2) * tsp_sparse_reward.num_cities

    # Check that all but the last step leads to 0 reward.
    next_state = state
    while not timestep.last():
        for action, is_valid in enumerate(timestep.observation.action_mask):
            if is_valid:
                next_state, timestep = step_fn(state, action)
                reward = sparse_reward(state, action, next_state, is_valid)
                if timestep.last():
                    # At the end of the episode, check that the reward is the negative tour length.
                    sorted_cities = state.coordinates[next_state.trajectory]
                    sorted_cities_rolled = jnp.roll(sorted_cities, 1, axis=0)
                    tour_length = jnp.linalg.norm(
                        sorted_cities - sorted_cities_rolled, axis=-1
                    ).sum()
                    assert reward == -tour_length
                else:
                    # Check that the reward is 0 for every non-final valid action.
                    assert reward == 0
            else:
                # Check that a penalty is given for every invalid action.
                invalid_next_state, _ = step_fn(state, action)
                reward = sparse_reward(state, action, invalid_next_state, is_valid)
                assert reward == penalty
        state = next_state
