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

from jumanji.environments.routing.connector.reward import DenseRewardFn
from jumanji.environments.routing.connector.types import State


def test_dense_reward(
    state: State, state1: State, state2: State, action1: chex.Array, action2: chex.Array
) -> None:
    timestep_reward = -0.03
    connected_reward = 0.1

    dense_reward_fn = jax.jit(
        DenseRewardFn(
            timestep_reward=timestep_reward,
            connected_reward=connected_reward,
        )
    )

    # Reward of moving between the same states should be 0.
    reward = dense_reward_fn(state, jnp.array([0, 0, 0]), state)
    chex.assert_rank(reward, 0)
    assert jnp.isclose(reward, jnp.asarray(timestep_reward * 3))

    # Reward for no agents finished to 2 agents finished.
    reward = dense_reward_fn(state, action1, state1)
    chex.assert_rank(reward, 0)
    expected_reward = connected_reward * 2 + timestep_reward * 3
    assert jnp.isclose(reward, expected_reward)

    # Reward for some agents finished to all agents finished.
    reward = dense_reward_fn(state1, action2, state2)
    chex.assert_rank(reward, 0)
    assert jnp.isclose(reward, jnp.array(connected_reward + timestep_reward))

    # Reward for none finished to all finished
    reward = dense_reward_fn(state, action1, state2)
    chex.assert_rank(reward, 0)
    assert jnp.isclose(reward, jnp.array((connected_reward + timestep_reward) * 3))

    # Reward of all finished to all finished.
    reward = dense_reward_fn(state2, jnp.zeros(3), state2)
    chex.assert_rank(reward, 0)
    assert jnp.isclose(reward, jnp.zeros(1))
