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

import jax.numpy as jnp

from jumanji.environments.swarms.search_and_rescue import reward


def test_rewards_from_found_targets() -> None:
    targets_found = jnp.array([[False, True, True], [False, False, True]], dtype=bool)

    shared_rewards = reward.SharedRewardFn()(targets_found, 0, 10)

    assert shared_rewards.shape == (2,)
    assert shared_rewards.dtype == jnp.float32
    assert jnp.allclose(shared_rewards, jnp.array([1.5, 0.5]))

    individual_rewards = reward.IndividualRewardFn()(targets_found, 0, 10)

    assert individual_rewards.shape == (2,)
    assert individual_rewards.dtype == jnp.float32
    assert jnp.allclose(individual_rewards, jnp.array([2.0, 1.0]))

    shared_scaled_rewards = reward.SharedScaledRewardFn()(targets_found, 0, 10)

    assert shared_scaled_rewards.shape == (2,)
    assert shared_scaled_rewards.dtype == jnp.float32
    assert jnp.allclose(shared_scaled_rewards, jnp.array([1.5, 0.5]))

    shared_scaled_rewards = reward.SharedScaledRewardFn()(targets_found, 10, 10)

    assert shared_scaled_rewards.shape == (2,)
    assert shared_scaled_rewards.dtype == jnp.float32
    assert jnp.allclose(shared_scaled_rewards, jnp.array([0.0, 0.0]))
