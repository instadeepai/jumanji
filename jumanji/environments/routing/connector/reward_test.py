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

from jumanji.environments.routing.connector.reward import (
    DenseRewardFn,
    SharedDenseRewardFn,
    SharedSparseRewardFn,
    SparseRewardFn,
)
from jumanji.environments.routing.connector.types import State


def test_dense_reward(
    state: State, state1: State, state2: State, action1: chex.Array, action2: chex.Array
) -> None:
    timestep_reward = -0.03
    connected_reward = 0.1
    num_agents = 3

    dense_reward_fn = jax.jit(
        DenseRewardFn(
            timestep_reward=timestep_reward,
            connected_reward=connected_reward,
        )
    )

    # Reward of moving between the same states.
    reward = dense_reward_fn(state, jnp.array([0, 0, 0]), state)
    chex.assert_rank(reward, 1)
    assert jnp.allclose(reward, jnp.array([timestep_reward] * num_agents))

    # Reward for no agents finished to 2 agents finished.
    reward = dense_reward_fn(state, action1, state1)
    chex.assert_rank(reward, 1)
    expected_reward = jnp.array([connected_reward, 0, connected_reward]) + timestep_reward
    assert jnp.allclose(reward, expected_reward)

    # Reward for some agents finished to all agents finished.
    reward = dense_reward_fn(state1, action2, state2)
    chex.assert_rank(reward, 1)
    expected_reward = jnp.array([0, connected_reward + timestep_reward, 0])
    assert jnp.allclose(reward, expected_reward)

    # Reward for none finished to all finished
    reward = dense_reward_fn(state, action1, state2)
    chex.assert_rank(reward, 1)
    assert jnp.allclose(reward, jnp.array([connected_reward + timestep_reward] * num_agents))

    # Reward of all finished to all finished.
    reward = dense_reward_fn(state2, jnp.zeros(num_agents), state2)
    chex.assert_rank(reward, 1)
    assert jnp.allclose(reward, jnp.zeros(1))


def test_shared_dense_reward(
    state: State, state1: State, state2: State, action1: chex.Array, action2: chex.Array
) -> None:
    timestep_reward = -0.03
    connected_reward = 0.1
    num_agents = 3

    reward_fn = jax.jit(
        SharedDenseRewardFn(
            timestep_reward=timestep_reward,
            connected_reward=connected_reward,
        )
    )

    # Reward of moving between the same state.
    reward = reward_fn(state, jnp.array([0, 0, 0]), state)
    chex.assert_rank(reward, 1)
    chex.assert_shape(reward, (num_agents,))
    assert jnp.allclose(reward, jnp.array([timestep_reward * num_agents] * num_agents))

    # Reward for no agents finished to 2 agents finished.
    reward = reward_fn(state, action1, state1)
    chex.assert_rank(reward, 1)
    chex.assert_shape(reward, (num_agents,))
    expected_reward = jnp.array(
        [(connected_reward * 2 + timestep_reward * num_agents)] * num_agents
    )
    assert jnp.allclose(reward, expected_reward)

    # Reward for some agents finished to all agents finished.
    reward = reward_fn(state1, action2, state2)
    chex.assert_rank(reward, 1)
    chex.assert_shape(reward, (num_agents,))
    expected_reward = jnp.array([connected_reward + timestep_reward] * num_agents)
    assert jnp.allclose(reward, expected_reward)

    # Reward for none finished to all finished
    reward = reward_fn(state, action1, state2)
    chex.assert_rank(reward, 1)
    chex.assert_shape(reward, (num_agents,))
    assert jnp.allclose(
        reward, jnp.array([(connected_reward + timestep_reward) * num_agents] * num_agents)
    )

    # Reward of all finished to all finished.
    reward = reward_fn(state2, jnp.zeros(num_agents), state2)
    chex.assert_rank(reward, 1)
    chex.assert_shape(reward, (num_agents,))
    assert jnp.allclose(reward, jnp.zeros(num_agents))


def test_sparse_reward(
    state: State, state1: State, state2: State, action1: chex.Array, action2: chex.Array
) -> None:
    num_agents = 3

    reward_fn = jax.jit(SparseRewardFn())

    # Reward of moving between the same states should be 0.
    reward = reward_fn(state, jnp.array([0, 0, 0]), state)
    chex.assert_rank(reward, 1)
    chex.assert_shape(reward, (num_agents,))
    assert jnp.allclose(reward, jnp.zeros(num_agents))

    # Reward for no agents finished to 2 agents finished.
    reward = reward_fn(state, action1, state1)
    chex.assert_rank(reward, 1)
    chex.assert_shape(reward, (num_agents,))
    assert jnp.allclose(reward, jnp.array([1, 0, 1]))

    # Reward for some agents finished to all agents finished.
    reward = reward_fn(state1, action2, state2)
    chex.assert_rank(reward, 1)
    chex.assert_shape(reward, (num_agents,))
    assert jnp.allclose(reward, jnp.array([0, 1, 0]))

    # Reward for none finished to all finished
    reward = reward_fn(state, action1, state2)
    chex.assert_rank(reward, 1)
    chex.assert_shape(reward, (num_agents,))
    assert jnp.allclose(reward, jnp.array([1, 1, 1]))

    # Reward of all finished to all finished.
    reward = reward_fn(state2, jnp.zeros(num_agents), state2)
    chex.assert_rank(reward, 1)
    chex.assert_shape(reward, (num_agents,))
    assert jnp.allclose(reward, jnp.zeros(num_agents))


def test_shared_sparse_reward(
    state: State, state1: State, state2: State, action1: chex.Array, action2: chex.Array
) -> None:
    num_agents = 3

    reward_fn = jax.jit(SharedSparseRewardFn())

    # Reward of moving between the same states should be 0.
    reward = reward_fn(state, jnp.array([0, 0, 0]), state)
    chex.assert_rank(reward, 1)
    chex.assert_shape(reward, (num_agents,))
    assert jnp.allclose(reward, jnp.zeros(num_agents))

    # Reward for no agents finished to 2 agents finished.
    reward = reward_fn(state, action1, state1)
    chex.assert_rank(reward, 1)
    chex.assert_shape(reward, (num_agents,))
    assert jnp.allclose(reward, jnp.array([2, 2, 2]))

    # Reward for some agents finished to all agents finished.
    reward = reward_fn(state1, action2, state2)
    chex.assert_rank(reward, 1)
    chex.assert_shape(reward, (num_agents,))
    assert jnp.allclose(reward, jnp.array([1, 1, 1]))

    # Reward for none finished to all finished
    reward = reward_fn(state, action1, state2)
    chex.assert_rank(reward, 1)
    chex.assert_shape(reward, (num_agents,))
    assert jnp.allclose(reward, jnp.array([3, 3, 3]))

    # Reward of all finished to all finished.
    reward = reward_fn(state2, jnp.zeros(num_agents), state2)
    chex.assert_rank(reward, 1)
    chex.assert_shape(reward, (num_agents,))
    assert jnp.allclose(reward, jnp.zeros(num_agents))
