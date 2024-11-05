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

from jumanji.environments.swarms.common.types import AgentParams
from jumanji.environments.swarms.common.updates import init_state
from jumanji.environments.swarms.predator_prey.types import State


class Generator(abc.ABC):
    def __init__(self, num_predators: int, num_prey: int) -> None:
        """Interface for instance generation for the `PredatorPrey` environment.

        Args:
            num_predators: Number of predator agents
            num_prey: Number of prey agents
        """
        self.num_predators = num_predators
        self.num_prey = num_prey

    @abc.abstractmethod
    def __call__(
        self, key: chex.PRNGKey, predator_params: AgentParams, prey_params: AgentParams
    ) -> State:
        """Generate initial agent positions and velocities.

        Args:
            key: random key.
            predator_params: Predator `AgentParams`.
            prey_params: Prey `AgentParams`.

        Returns:
            Initial agent `State`.
        """


class RandomGenerator(Generator):
    def __call__(
        self, key: chex.PRNGKey, predator_params: AgentParams, prey_params: AgentParams
    ) -> State:
        """Generate random initial agent positions and velocities.

        Args:
            key: random key.
            predator_params: Predator `AgentParams`.
            prey_params: Prey `AgentParams`.

        Returns:
            state: the generated state.
        """
        key, predator_key, prey_key = jax.random.split(key, num=3)
        predator_state = init_state(self.num_predators, predator_params, predator_key)
        prey_state = init_state(self.num_prey, prey_params, prey_key)
        state = State(predators=predator_state, prey=prey_state, key=key)
        return state
