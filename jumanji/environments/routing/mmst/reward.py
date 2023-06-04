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
import jax
import jax.numpy as jnp

from jumanji.environments.routing.mmst.constants import (
    INVALID_CHOICE,
    INVALID_TIE_BREAK,
)
from jumanji.environments.routing.mmst.types import State


class RewardFn(abc.ABC):
    """Abstract class for `MMST` rewards."""

    @abc.abstractmethod
    def __call__(
        self, state: State, actions: chex.Array, nodes_to_connect: chex.Array
    ) -> chex.Array:
        """The reward function used in the `MMST` environment.

        Args:
            state: Environment state
            actions: Actions taken by all the agents to reach this state.
            nodes_to_connect: Array containing the nodes each agent needs to connect.
        Returns:
            reward
        """


class DenseRewardFn(RewardFn):
    """Dense reward function."""

    def __init__(self, reward_values: chex.Array) -> None:
        """Instantiates the dense reward function.

        Args:
            reward_values: array with rewards for each type of event.
              This is a list with 3 values. The first is the reward for
              connecting a node, the second is the reward for a non connection
              and the third is the reward for an invalid option.
        """

        self._reward_connected = reward_values[0]
        self._reward_time_step = reward_values[1]
        self._reward_noop = reward_values[2]

        def reward_fun(nodes: chex.Array, action: int, node: int) -> jnp.float_:
            noop_coeff = action == INVALID_CHOICE
            same_coeff = action == INVALID_TIE_BREAK

            is_connection = jnp.isin(node, nodes) & (action > INVALID_CHOICE)

            return jax.lax.cond(
                is_connection,
                lambda: self._reward_connected - same_coeff * self._reward_connected,
                lambda: self._reward_time_step
                + noop_coeff * self._reward_noop
                - same_coeff * self._reward_time_step,
            )

        self.reward_fun = reward_fun

    def __call__(
        self, state: State, actions: chex.Array, nodes_to_connect: chex.Array
    ) -> chex.Array:

        num_agents = len(actions)

        rewards = jnp.zeros((num_agents,), dtype=jnp.float32)
        for agent in range(num_agents):
            reward_i = self.reward_fun(
                nodes_to_connect[agent],
                actions[agent],
                state.positions[agent],
            )
            rewards = rewards.at[agent].set(reward_i)

        rewards *= ~state.finished_agents
        return jnp.sum(rewards)
