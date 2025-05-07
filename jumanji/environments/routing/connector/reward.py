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

import abc

import chex
import jax.numpy as jnp
from typing_extensions import override

from jumanji.environments.routing.connector.types import State


class RewardFn(abc.ABC):
    """Abstract class for `Connector` rewards."""

    @abc.abstractmethod
    def __call__(
        self,
        state: State,
        action: chex.Array,
        next_state: State,
    ) -> chex.Array:
        """The reward function used in the `Connector` environment.

        Args:
            state: `Connector` state before taking an action.
            action: action taken from `state` to reach `next_state`.
            next_state: `Connector` state after taking the action.

        Returns:
            The reward for the current step.
        """


class DenseRewardFn(RewardFn):
    """Returns: reward of 1.0 for each agent that connects on that step and adds a penalty of
    -0.03, per agent, at every timestep where they have yet to connect.

    The reward is of shape num_agents where a value in dimension 1 corresponds agent 1's reward.
    """

    def __init__(self, connected_reward: float = 1.0, timestep_reward: float = -0.03) -> None:
        """Instantiates a dense reward function for the `Connector` environment.

        Args:
            num_agents: the number of agents in the environment.
            connected_reward: reward agent if it connects on that step.
            timestep_reward: reward penalty for every timestep, encourages agent to connect quickly.
        """
        self.timestep_reward = timestep_reward
        self.connected_reward = connected_reward

    def __call__(
        self,
        state: State,
        action: chex.Array,
        next_state: State,
    ) -> chex.Array:
        connected_rewards = self.connected_reward * jnp.asarray(
            ~state.agents.connected & next_state.agents.connected, jnp.float32
        )
        timestep_rewards = self.timestep_reward * jnp.asarray(~state.agents.connected, jnp.float32)
        return connected_rewards + timestep_rewards


class SharedDenseRewardFn(DenseRewardFn):
    """Returns: the sum of agents of the `MultiAgentDenseRewardFn`."""

    @override
    def __call__(self, state: State, action: chex.Array, next_state: State) -> chex.Array:
        individual_reward = super().__call__(state, action, next_state)
        num_agents = individual_reward.shape[0]
        return jnp.sum(individual_reward, axis=-1, keepdims=True).repeat(num_agents, axis=-1)


class SparseRewardFn(RewardFn):
    """Returns: a reward of 1 for each agent that connects on the current step.

    The reward is of shape num_agents where a value in dimension 1 corresponds agent 1's reward.
    """

    @override
    def __call__(self, state: State, action: chex.Array, next_state: State) -> chex.Array:
        return jnp.asarray(~state.agents.connected & next_state.agents.connected, jnp.float32)


class SharedSparseRewardFn(SparseRewardFn):
    """Returns: a reward of 1 for any agent that connects on the current step."""

    def __init__(self) -> None: ...

    @override
    def __call__(self, state: State, action: chex.Array, next_state: State) -> chex.Array:
        individual_reward = super().__call__(state, action, next_state)
        num_agents = individual_reward.shape[0]
        return jnp.sum(individual_reward, axis=-1, keepdims=True).repeat(num_agents, axis=-1)
