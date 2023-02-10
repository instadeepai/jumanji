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

from jumanji.environments.routing.connector.constants import NOOP
from jumanji.environments.routing.connector.types import State


class RewardFn(abc.ABC):
    """Abstract class for Connector rewards."""

    @abc.abstractmethod
    def __call__(
        self,
        state: State,
        next_state: State,
        action: chex.Array,
    ) -> chex.Array:
        """The reward function used in the Connector environment.

        Args:
            state: connector state before taking an action.
            next_state: connector state after taking the action.
            action: action: multi-dimensional action taken to reach this state.
            step: how many steps have passed this episode.
            reward_config_kwargs: default arguments for configuring the reward.

        Returns:
            The reward for the current step.
        """


class SparseRewardFn(RewardFn):
    """Rewards each agent with 1 if the agent connected on that step, otherwise 0."""

    def __call__(
        self, state: State, next_state: State, action: chex.Array
    ) -> chex.Array:
        del action
        return jnp.asarray(~state.agents.connected & next_state.agents.connected, float)


class DenseRewardFn(RewardFn):
    """Rewards: `timestep_reward` each timestep, `connected_reward` if it connects and `noop_reward`
    for each noop.

    Each agent receives a separate reward and rewards are 0 after an agent has connected.
    """

    def __init__(
        self,
        timestep_reward: float = -0.03,
        connected_reward: float = 0.1,
        noop_reward: float = -0.01,
    ) -> None:
        """Dense reward function initialiser: sets reward scales.

        Args:
            timestep_reward: reward every timestep, encourages agent to connect quickly.
            connected_reward: reward agent when connected.
            noop_reward: reward agent negatively when doing a noop to discourage the action.
        """
        self.timestep_reward = timestep_reward
        self.connected_reward = connected_reward
        self.noop_reward = noop_reward

    def __call__(
        self,
        state: State,
        next_state: State,
        action: chex.Array,
    ) -> chex.Array:
        connected_reward = self.connected_reward * next_state.agents.connected
        noop_reward = self.noop_reward * (action == NOOP)
        agents_rewards = jnp.asarray(
            connected_reward + noop_reward + self.timestep_reward, dtype=float
        )
        return jnp.where(state.agents.connected, 0.0, agents_rewards)
