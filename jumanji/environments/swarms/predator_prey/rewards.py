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
from typing import Union

import chex
import jax.numpy as jnp
from esquilax.transforms import nearest_neighbour, spatial
from esquilax.utils import shortest_distance

from jumanji.environments.swarms.predator_prey.types import Rewards, State


class RewardFn(abc.ABC):
    """Abstract class for `PredatorPrey` rewards."""

    @abc.abstractmethod
    def __call__(self, state: State) -> Rewards:
        """The reward function used in the `PredatorPrey` environment.

        Args:
            state: `PredatorPrey` state.

        Returns:
            The reward for the current step for individual agents.
        """


class SparseRewards(RewardFn):
    """Sparse rewards applied when agents come into contact.

    Rewards applied when predators and prey come into contact
    (i.e. overlap), positively rewarding predators and negatively
    penalising prey. Attempts to model predators `capturing` prey.
    """

    def __init__(self, agent_radius: float, predator_reward: float, prey_penalty: float) -> None:
        """
        Initialise a sparse reward function.

        Args:
            agent_radius: Radius of simulated agents.
            predator_reward: Predator reward value.
            prey_penalty: Prey penalty (this is negated when applied).
        """
        self.agent_radius = agent_radius
        self.prey_penalty = prey_penalty
        self.predator_reward = predator_reward

    def prey_rewards(
        self,
        _key: chex.PRNGKey,
        _params: None,
        _prey: None,
        _predator: None,
    ) -> float:
        """Penalise a prey agent if contacted by a predator agent.

        Apply a negative penalty to prey agents that collide
        with a prey agent. This function is applied using an
        Esquilax spatial interaction which accumulates rewards.

        Args:
            _key: Dummy JAX random key .
            _params: Dummy params (required by Esquilax).
            _prey: Dummy agent-state (required by Esquilax).
            _predator: Dummy agent-state (required by Esquilax).

        Returns:
            float: Negative penalty applied to prey agent.
        """
        return -self.prey_penalty

    def predator_rewards(
        self,
        _key: chex.PRNGKey,
        _params: None,
        _predator: None,
        _prey: None,
    ) -> float:
        """Reward a predator agent if it is within range of a prey agent
        (required by Esquilax)
                Apply a fixed positive reward if a predator agent is within
                a fixed range of a prey-agent. This function can
                be used with an Esquilax spatial interaction to
                apply rewards to agents in range.

                Args:
                    _key: Dummy JAX random key (required by Esquilax).
                    _params: Dummy params (required by Esquilax).
                    _prey: Dummy agent-state (required by Esquilax).
                    _predator: Dummy agent-state (required by Esquilax).

                Returns:
                    float: Predator agent reward.
        """
        return self.predator_reward

    def __call__(self, state: State) -> Rewards:
        prey_rewards = spatial(
            self.prey_rewards,
            reduction=jnp.add,
            default=0.0,
            include_self=False,
            i_range=2 * self.agent_radius,
        )(
            state.key,
            None,
            None,
            None,
            pos=state.prey.pos,
            pos_b=state.predators.pos,
        )
        predator_rewards = nearest_neighbour(
            self.predator_rewards,
            default=0.0,
            i_range=2 * self.agent_radius,
        )(
            state.key,
            None,
            None,
            None,
            pos=state.predators.pos,
            pos_b=state.prey.pos,
        )
        return Rewards(
            predators=predator_rewards,
            prey=prey_rewards,
        )


class DistanceRewards(RewardFn):
    """Rewards based on proximity to other agents.

    Rewards generated based on an agents proximity to other
    agents within their vision range. Predator rewards increase
    as they get closer to prey, and prey rewards become
    increasingly negative as they get closer to predators.
    Rewards are summed over all other agents within range of
    an agent.
    """

    def __init__(
        self,
        predator_vision_range: float,
        prey_vision_range: float,
        predator_reward: float,
        prey_penalty: float,
    ) -> None:
        """
        Initialise a distance reward function

        Args:
            predator_vision_range: Predator agent vision range.
            prey_vision_range: Prey agent vision range.
            predator_reward: Max reward value applied to
              predator agents.
            prey_penalty: Max reward value applied to prey agents
              (this value is negated when applied).
        """
        self.predator_vision_range = predator_vision_range
        self.prey_vision_range = prey_vision_range
        self.prey_penalty = prey_penalty
        self.predator_reward = predator_reward

    def prey_rewards(
        self,
        _key: chex.PRNGKey,
        _params: None,
        prey_pos: chex.Array,
        predator_pos: chex.Array,
        *,
        i_range: float,
    ) -> Union[float, chex.Array]:
        """Penalise a prey agent based on distance from a predator agent.

        Apply a negative penalty based on a distance between
        agents. The penalty is a linear function of distance,
        0 at max distance up to `-penalty` at 0 distance. This function
        can be used with an Esquilax spatial interaction to accumulate
        rewards between agents.

        Args:
            _key: Dummy JAX random key (required by Esquilax).
            _params: Dummy params (required by Esquilax).
            prey_pos: Prey positions.
            predator_pos: Predator positions.
            i_range: Static interaction range.

        Returns:
            float: Agent rewards.
        """
        d = shortest_distance(prey_pos, predator_pos) / i_range
        return self.prey_penalty * (d - 1.0)

    def predator_rewards(
        self,
        _key: chex.PRNGKey,
        _params: None,
        predator_pos: chex.Array,
        prey_pos: chex.Array,
        *,
        i_range: float,
    ) -> Union[float, chex.Array]:
        """Reward a predator agent based on distance from a prey agent.

        Apply a positive reward based on the linear distance between
        a predator and prey agent. Rewards are zero at the max
        interaction distance, and maximal at 0 range. This function
        can be used with an Esquilax spatial interaction to accumulate
        rewards between agents.

        Args:
            _key: Dummy JAX random key (required by Esquilax).
            _params: Dummy parameters (required by Esquilax).
            predator_pos: Predator position.
            prey_pos: Prey position.
            i_range: Static interaction range.

        Returns:
            float: Predator agent reward.
        """
        d = shortest_distance(predator_pos, prey_pos) / i_range
        return self.predator_reward * (1.0 - d)

    def __call__(self, state: State) -> Rewards:
        prey_rewards = spatial(
            self.prey_rewards,
            reduction=jnp.add,
            default=0.0,
            include_self=False,
            i_range=self.prey_vision_range,
        )(
            state.key,
            None,
            state.prey.pos,
            state.predators.pos,
            pos=state.prey.pos,
            pos_b=state.predators.pos,
            i_range=self.prey_vision_range,
        )
        predator_rewards = spatial(
            self.predator_rewards,
            reduction=jnp.add,
            default=0.0,
            include_self=False,
            i_range=self.predator_vision_range,
        )(
            state.key,
            None,
            state.predators.pos,
            state.prey.pos,
            pos=state.predators.pos,
            pos_b=state.prey.pos,
            i_range=self.prey_vision_range,
        )

        return Rewards(
            predators=predator_rewards,
            prey=prey_rewards,
        )
