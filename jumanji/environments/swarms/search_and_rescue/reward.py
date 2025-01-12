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


class RewardFn(abc.ABC):
    """Abstract class for `SearchAndRescue` rewards."""

    @abc.abstractmethod
    def __call__(self, found_targets: chex.Array, step: int, time_limit: int) -> chex.Array:
        """The reward function used in the `SearchAndRescue` environment.

        Args:
            found_targets: Array of boolean flags indicating if an
            agent has found a target.

        Returns:
            Individual reward for each agent.
        """


def _normalise_rewards(rewards: chex.Array) -> chex.Array:
    norms = jnp.sum(rewards, axis=0)[jnp.newaxis]
    rewards = jnp.where(norms > 0, rewards / norms, rewards)
    return rewards


def _scale_rewards(rewards: chex.Array, step: int, time_limit: int) -> chex.Array:
    scale = (time_limit - step) / time_limit
    return scale * rewards


class SharedRewardFn(RewardFn):
    """
    Calculate per agent rewards from detected targets

    Targets detected by multiple agents share rewards. Agents
    can receive rewards for detecting multiple targets.
    """

    def __call__(self, found_targets: chex.Array, step: int, time_limit: int) -> chex.Array:
        rewards = found_targets.astype(float)
        rewards = _normalise_rewards(rewards)
        rewards = jnp.sum(rewards, axis=1)
        return rewards


class SharedScaledRewardFn(RewardFn):
    """
    Calculate per agent rewards from detected targets and scale by timestep

    Targets detected by multiple agents share rewards. Agents
    can receive rewards for detecting multiple targets.
    Rewards are linearly scaled by the current time step such that
    rewards received are 0 at the final step.
    """

    def __call__(self, found_targets: chex.Array, step: int, time_limit: int) -> chex.Array:
        rewards = found_targets.astype(float)
        rewards = _normalise_rewards(rewards)
        rewards = jnp.sum(rewards, axis=1)
        rewards = _scale_rewards(rewards, step, time_limit)
        return rewards


class IndividualRewardFn(RewardFn):
    """
    Calculate per agent rewards from detected targets

    Each agent that detects a target receives a +1 reward
    even if a target is detected by multiple agents.
    """

    def __call__(self, found_targets: chex.Array, step: int, time_limit: int) -> chex.Array:
        rewards = found_targets.astype(float)
        rewards = jnp.sum(rewards, axis=1)
        return rewards


class IndividualScaledRewardFn(RewardFn):
    """
    Calculate per agent rewards from detected targets and scale by timestep

    Each agent that detects a target receives a +1 reward
    even if a target is detected by multiple agents.
    Rewards are linearly scaled by the current time step such that
    rewards received are 0 at the final step.
    """

    def __call__(self, found_targets: chex.Array, step: int, time_limit: int) -> chex.Array:
        rewards = found_targets.astype(float)
        rewards = jnp.sum(rewards, axis=1)
        rewards = _scale_rewards(rewards, step, time_limit)
        return rewards
