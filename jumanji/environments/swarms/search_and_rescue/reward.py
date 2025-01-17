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


class IndividualRewardFn(RewardFn):
    """
    Calculate individual agent rewards from detected targets inside a step

    Assigns individual rewards to each agent based on the number of newly
    found targets inside a step.

    Note that multiple agents can locate the same target inside a single
    step. By default, rewards are split between the locating agents and
    rewards linearly decrease over time (i.e. rewards are +1 per target in
    the first step, decreasing to 0 at the final step).
    """

    def __init__(self, split_rewards: bool = True, scale_rewards: bool = True):
        """
        Initialise reward-function

        Args:
            split_rewards: If ``True`` rewards will be split in the case multiple agents
                find a target at the same time. If ``False`` then agents will receive the
                full reward irrespective of the number of agents who located the
                target simultaneously.
            scale_rewards: If ``True`` rewards granted will linearly decrease over time from
                +1 per target to 0 at the final step. If ``False`` rewards will be fixed at
                +1 per target irrespective of step.
        """
        self.split_rewards = split_rewards
        self.scale_rewards = scale_rewards

    def __call__(self, found_targets: chex.Array, step: int, time_limit: int) -> chex.Array:
        rewards = found_targets.astype(float)

        if self.split_rewards:
            rewards = _normalise_rewards(rewards)

        rewards = jnp.sum(rewards, axis=1)

        if self.scale_rewards:
            rewards = _scale_rewards(rewards, step, time_limit)

        return rewards
