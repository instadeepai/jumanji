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

from jumanji.environments.swarms.common.types import AgentParams
from jumanji.environments.swarms.common.updates import init_state
from jumanji.environments.swarms.search_and_rescue.types import State, TargetState


class Generator(abc.ABC):
    def __init__(self, num_searchers: int, num_targets: int) -> None:
        """Interface for instance generation for the `SearchAndRescue` environment.

        Args:
            num_searchers: Number of searcher agents
            num_targets: Number of search targets
        """
        self.num_searchers = num_searchers
        self.num_targets = num_targets

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey, searcher_params: AgentParams) -> State:
        """Generate initial agent positions and velocities.

        Args:
            key: random key.
            searcher_params: Searcher `AgentParams`.

        Returns:
            Initial agent `State`.
        """


class RandomGenerator(Generator):
    def __call__(self, key: chex.PRNGKey, searcher_params: AgentParams) -> State:
        """Generate random initial agent positions and velocities, and random target positions.

        Args:
            key: random key.
            searcher_params: Searcher `AgentParams`.

        Returns:
            state: the generated state.
        """
        key, searcher_key, target_key = jax.random.split(key, num=3)
        searcher_state = init_state(self.num_searchers, searcher_params, searcher_key)
        target_pos = jax.random.uniform(target_key, (self.num_targets, 2))
        state = State(
            searchers=searcher_state,
            targets=TargetState(
                pos=target_pos, found=jnp.full((self.num_targets,), False, dtype=bool)
            ),
            key=key,
        )
        return state
