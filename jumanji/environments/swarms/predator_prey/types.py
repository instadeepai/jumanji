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
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex

from jumanji.environments.swarms.common.types import AgentState


@dataclass
class State:
    """
    predators: Predator agent states.
    prey: Prey agent states.
    key: JAX random key.
    step: Environment step number
    """

    predators: AgentState
    prey: AgentState
    key: chex.PRNGKey
    step: int = 0


@dataclass
class Actions:
    """
    predators: Array of actions for predator agents.
    prey: Array of actions for prey agents.
    """

    predators: chex.Array  # (num_predators, 2)
    prey: chex.Array  # (num_prey, 2)


@dataclass
class Rewards:
    """
    predators: Array of individual rewards for predator agents.
    prey: Array of individual rewards for prey agents.
    """

    predators: chex.Array  # (num_predators,)
    prey: chex.Array  # (num_prey,)


class Observation(NamedTuple):
    """
    Individual observations for predator and prey agents.

    Each agent generates an independent observation, an array of
    values representing the distance along a ray from the agent to
    the nearest neighbour, with each cell representing a ray angle
    (with `num_vision` rays evenly distributed over the agents
    field of vision). Prey and prey agent types are visualised
    independently to allow agents to observe both local position and type.

    For example if a prey agent sees a predator straight ahead and
    `num_vision = 5` then the observation array could be

    ```
    [1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ```

    or if it observes another prey agent

    ```
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0]
    ```

    where `1.0` indicates there is no agents along that ray,
    and `0.5` is the normalised distance to the other agent.

    - `predators`: jax array (float) of shape `(num_predators, 2 * num_vision)`
      in the unit interval.
    - `prey`: jax array (float) of shape `(num_prey, 2 * num_vision)` in the
      unit interval.
    """

    predators: chex.Array  # (num_predators, num_vision)
    prey: chex.Array  # (num_prey, num_vision)
