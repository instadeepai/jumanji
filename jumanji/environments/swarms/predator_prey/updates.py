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
    return -penalty


def distance_prey_rewards(
    _k: chex.PRNGKey,
    penalty: float,
    prey: types.AgentState,
    predator: types.AgentState,
    *,
    i_range: float,
) -> Union[float, chex.Array]:
    d = esquilax.utils.shortest_distance(prey.pos, predator.pos) / i_range
    return penalty * (d - 1.0)


def sparse_predator_rewards(
    _k: chex.PRNGKey,
    reward: float,
    _a: Optional[types.AgentState],
    _b: Optional[types.AgentState],
) -> float:
    return reward


def distance_predator_rewards(
    _k: chex.PRNGKey,
    reward: float,
    predator: types.AgentState,
    prey: types.AgentState,
    *,
    i_range: float,
) -> Union[float, chex.Array]:
    d = esquilax.utils.shortest_distance(predator.pos, prey.pos) / i_range
    return reward * (1.0 - d)
