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
import jax.numpy as jnp
import pytest

from jumanji.environments.swarms.search_and_rescue import reward


@pytest.fixture
def target_states() -> chex.Array:
    return jnp.array([[False, True, True], [False, False, True]], dtype=bool)


def test_shared_rewards(target_states: chex.Array) -> None:
    shared_rewards = reward.SharedRewardFn()(target_states, 0, 10)

    assert shared_rewards.shape == (2,)
    assert shared_rewards.dtype == jnp.float32
    assert jnp.allclose(shared_rewards, jnp.array([1.5, 0.5]))


def test_individual_rewards(target_states: chex.Array) -> None:
    individual_rewards = reward.IndividualRewardFn()(target_states, 0, 10)

    assert individual_rewards.shape == (2,)
    assert individual_rewards.dtype == jnp.float32
    assert jnp.allclose(individual_rewards, jnp.array([2.0, 1.0]))


def test_shared_scaled_rewards(target_states: chex.Array) -> None:
    reward_fn = reward.SharedScaledRewardFn()

    shared_scaled_rewards = reward_fn(target_states, 0, 10)

    assert shared_scaled_rewards.shape == (2,)
    assert shared_scaled_rewards.dtype == jnp.float32
    assert jnp.allclose(shared_scaled_rewards, jnp.array([1.5, 0.5]))

    shared_scaled_rewards = reward_fn(target_states, 10, 10)

    assert shared_scaled_rewards.shape == (2,)
    assert shared_scaled_rewards.dtype == jnp.float32
    assert jnp.allclose(shared_scaled_rewards, jnp.array([0.0, 0.0]))


def test_individual_scaled_rewards(target_states: chex.Array) -> None:
    reward_fn = reward.IndividualScaledRewardFn()

    individual_scaled_rewards = reward_fn(target_states, 0, 10)

    assert individual_scaled_rewards.shape == (2,)
    assert individual_scaled_rewards.dtype == jnp.float32
    assert jnp.allclose(individual_scaled_rewards, jnp.array([2.0, 1.0]))

    individual_scaled_rewards = reward_fn(target_states, 10, 10)

    assert individual_scaled_rewards.shape == (2,)
    assert individual_scaled_rewards.dtype == jnp.float32
    assert jnp.allclose(individual_scaled_rewards, jnp.array([0.0, 0.0]))
