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
from typing import TYPE_CHECKING

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
class Observation:
    """
    predators: Local view of predator agents.
    prey: Local view of prey agents.
    """

    predators: chex.Array
    prey: chex.Array


@dataclass
class Actions:
    """
    predators: Array of actions for predator agents.
    prey: Array of actions for prey agents.
    """

    predators: chex.Array
    prey: chex.Array


@dataclass
class Rewards:
    """
    predators: Array of individual rewards for predator agents.
    prey: Array of individual rewards for prey agents.
    """

    predators: chex.Array
    prey: chex.Array
