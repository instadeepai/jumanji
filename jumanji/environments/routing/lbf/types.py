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

import chex
import jax.numpy as jnp

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class Entity:
    """
    id: unique number representing only this food.
    position: the position of this food.
    level: the level of this food.
    """

    id: chex.Array  # ()
    position: chex.Array  # (2,)
    level: chex.Array  # ()


@dataclass
class Agent(Entity):
    loading: chex.Array = jnp.asarray(False)  # () - bool: is loading food


@dataclass
class Food(Entity):
    eaten: chex.Array = jnp.asarray(False)  # () - bool: has been eaten


@dataclass
class State:
    """
    step_count: the index of the current step.
    agents: a stacked pytree of type Agent.
    key: random key used for auto-reset.
    """

    step_count: chex.Array  # ()
    agents: Agent  # (num_agents, ...)
    foods: Food  # (num_foods, ...)
    key: chex.PRNGKey  # (2,)


class Observation(NamedTuple):
    """
    The observation returned by the LBF environment.

    agent_views: (num_agents, grid_size, grid_size) int8 array representing the view of each agent.
    action_mask: boolean array representing whether each of the 5 actions is legal, for each agent.
    step_count: (int32) the current episode step.
    """

    agents_view: chex.Array  # (num_agents, grid_size, grid_size)
    action_mask: chex.Array  # (num_agents, 6)
    step_count: chex.Array  # ()
