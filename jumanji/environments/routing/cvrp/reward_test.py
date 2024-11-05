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

from jumanji.environments.routing.cvrp.env import CVRP
from jumanji.environments.routing.cvrp.reward import (
    DenseReward,
    SparseReward,
    compute_tour_length,
)


def test_sparse_reward__compute_tour_length() -> None:
    """Checks that the tour lengths are properly computed."""
    coordinates = jnp.array(
        [
            [0.65948975, 0.8527372],
            [0.18317401, 0.06975579],
            [0.4064678, 0.19167936],
            [0.92129254, 0.27006388],
            [0.7105516, 0.9370967],
            [0.5277389, 0.18168604],
            [0.47508526, 0.19661963],
            [0.46782017, 0.6201354],
            [0.4211073, 0.5530877],
            [0.94237375, 0.64736927],
            [0.97507954, 0.43589878],
        ]
    )

    trajectory = jnp.array([0, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 0, 6, 0, 0, 0, 0, 0])
    tour_length = compute_tour_length(coordinates, trajectory)
    assert jnp.isclose(tour_length, 6.8649917)

    trajectory = jnp.array([0, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 0, 6])
    assert jnp.isclose(compute_tour_length(coordinates, trajectory), 6.8649917)

    trajectory = jnp.array([0, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 0, 6, 0])
    assert jnp.isclose(compute_tour_length(coordinates, trajectory), 6.8649917)


def test_dense_reward(cvrp_dense_reward: CVRP, dense_reward: DenseReward) -> None:
    dense_reward = jax.jit(dense_reward)
    step_fn = jax.jit(cvrp_dense_reward.step)
    state, timestep = cvrp_dense_reward.reset(jax.random.PRNGKey(0))

    # Check that the reward is correct for the next node.
    state, timestep = step_fn(state, 0)
    for action in range(1, cvrp_dense_reward.num_nodes + 1):
        next_state, _ = step_fn(state, action)
        depot = state.coordinates[0]
        new_city = state.coordinates[action]
        distance = jnp.linalg.norm(new_city - depot)
        assert dense_reward(state, action, next_state, is_valid=True) == -distance

    # Check the reward for invalid action.
    next_state, _ = step_fn(state, 0)
    penalty = -jnp.sqrt(2) * 2 * cvrp_dense_reward.num_nodes
    assert dense_reward(state, 0, next_state, is_valid=False) == penalty


def test_sparse_reward(cvrp_sparse_reward: CVRP, sparse_reward: SparseReward) -> None:
    sparse_reward = jax.jit(sparse_reward)
    step_fn = jax.jit(cvrp_sparse_reward.step)
    state, timestep = cvrp_sparse_reward.reset(jax.random.PRNGKey(0))
    penalty = -jnp.sqrt(2) * 2 * cvrp_sparse_reward.num_nodes

    # Check that all but the last step leads to 0 reward.
    next_state = state
    while not timestep.last():
        for action, is_valid in enumerate(timestep.observation.action_mask):
            if is_valid:
                next_state, timestep = step_fn(state, action)
                if timestep.last():
                    # At the end of the episode, check that the reward is the negative tour length.
                    sorted_cities = state.coordinates[next_state.trajectory]
                    sorted_cities_rolled = jnp.roll(sorted_cities, 1, axis=0)
                    tour_length = jnp.linalg.norm(
                        sorted_cities - sorted_cities_rolled, axis=-1
                    ).sum()
                    reward = sparse_reward(state, action, next_state, is_valid)
                    assert jnp.isclose(reward, -tour_length)
                else:
                    # Check that the reward is 0 for every non-final valid action.
                    assert sparse_reward(state, action, next_state, is_valid) == 0
            else:
                # Check that a penalty is given for every invalid action.
                invalid_next_state, _ = step_fn(state, action)
                assert sparse_reward(state, action, invalid_next_state, is_valid) == penalty
        state = next_state
