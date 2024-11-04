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

from typing import Optional, Union

import chex
import esquilax

from jumanji.environments.swarms.common import types


def sparse_prey_rewards(
    _k: chex.PRNGKey,
    penalty: float,
    _prey: Optional[types.AgentState],
    _predator: Optional[types.AgentState],
) -> float:
    """Penalise a prey agent if contacted by a predator agent.

    Apply a negative penalty to prey agents that collide
    with a prey agent. This function is applied using an
    Esquilax spatial interaction.

    Args:
        _k: Dummy JAX random key.
        penalty: Penalty value.
        _prey: Optional unused prey agent-state.
        _predator: Optional unused predator agent-state.

    Returns:
        float: Negative penalty applied to prey agent.
    """
    return -penalty


def distance_prey_rewards(
    _k: chex.PRNGKey,
    penalty: float,
    prey: types.AgentState,
    predator: types.AgentState,
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
        _k: Dummy JAX random key.
        penalty: Maximum penalty applied.
        prey: Prey agent-state.
        predator: Predator agent-state.
        i_range: Static interaction range.

    Returns:
        float: Agent rewards.
    """
    d = esquilax.utils.shortest_distance(prey.pos, predator.pos) / i_range
    return penalty * (d - 1.0)


def sparse_predator_rewards(
    _k: chex.PRNGKey,
    reward: float,
    _predator: Optional[types.AgentState],
    _prey: Optional[types.AgentState],
) -> float:
    """Reward a predator agent if it is within range of a prey agent

    Apply a fixed positive reward if a predator agent is within
    a fixed range of a prey-agent. This function can
    be used with an Esquilax spatial interaction to
    apply rewards to agents in range.

    Args:
        _k: Dummy JAX random key.
        reward: Reward value to apply.
        _predator: Optional unused agent-state.
        _prey: Optional unused agent-state.

    Returns:
        float: Predator agent reward.
    """
    return reward


def distance_predator_rewards(
    _k: chex.PRNGKey,
    reward: float,
    predator: types.AgentState,
    prey: types.AgentState,
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
        _k: Dummy JAX random key.
        reward: Maximum reward value.
        predator: Predator agent-state.
        prey: Prey agent-state.
        i_range: Static interaction range.

    Returns:
        float@ Predator agent reward.
    """
    d = esquilax.utils.shortest_distance(predator.pos, prey.pos) / i_range
    return reward * (1.0 - d)
