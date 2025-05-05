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
from typing import Tuple, override

import chex
import jax.numpy as jnp

from jumanji import specs
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

    @property
    @abc.abstractmethod
    def shape(self) -> Tuple[int, ...]: ...

    def spec(self) -> specs.Array:
        return specs.Array(shape=self.shape, dtype=jnp.float32, name="reward")


class MultiAgentDenseRewardFn(RewardFn):
    """Returns: reward of 1.0 for each agent that connects on that step and adds a penalty of
    -0.03, per agent, at every timestep where they have yet to connect.

    The reward is of shape num_agents where a value in dimension 1 corresponds agent 1's reward.
    """

    def __init__(
        self, num_agents: int, connected_reward: float = 1.0, timestep_reward: float = -0.03
    ) -> None:
        """Instantiates a dense reward function for the `Connector` environment.

        Args:
            num_agents: the number of agents in the environment.
            connected_reward: reward agent if it connects on that step.
            timestep_reward: reward penalty for every timestep, encourages agent to connect quickly.
        """
        self.num_agents = num_agents
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

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self.num_agents,)


class SingleAgentDenseRewardFn(MultiAgentDenseRewardFn):
    """Returns: the sum of agents of the `MultiAgentDenseRewardFn`.

    The reward is a scalar, so all agents get the same (shared) reward.
    """

    def __init__(self) -> None: ...

    @override
    def __call__(self, state: State, action: chex.Array, next_state: State) -> chex.Array:
        return jnp.sum(super().__call__(state, action, next_state), axis=-1)

    @property
    def shape(self) -> Tuple[int, ...]:
        return ()


class MultiAgentSparseRewardFn(RewardFn):
    """Returns: a reward of 1 for each agent that connects on the current step.

    The reward is of shape num_agents where a value in dimension 1 corresponds agent 1's reward.
    """

    def __init__(self, num_agents: int) -> None:
        self.num_agents = num_agents

    @override
    def __call__(self, state: State, action: chex.Array, next_state: State) -> chex.Array:
        return jnp.asarray(~state.agents.connected & next_state.agents.connected, jnp.float32)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self.num_agents,)


class SingleAgentSparseRewardFn(MultiAgentSparseRewardFn):
    """Returns: a reward of 1 for any agent that connects on the current step.

    The reward is a scalar, so all agents get the same (shared) reward.
    """

    def __init__(self) -> None: ...

    @override
    def __call__(self, state: State, action: chex.Array, next_state: State) -> chex.Array:
        return jnp.sum(super().__call__(state, action, next_state), axis=-1)

    @property
    def shape(self) -> Tuple[int, ...]:
        return ()
