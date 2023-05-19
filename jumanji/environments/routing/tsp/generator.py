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

from jumanji.environments.routing.tsp.types import State


class Generator(abc.ABC):
    """Defines the abstract `Generator` base class. A `Generator` is responsible
    for generating a problem instance when the environment is reset.
    """

    def __init__(self, num_cities: int):
        """Abstract class implementing the attribute `num_cities`.

        Args:
            num_cities (int): the number of cities in the problem instance.
        """
        self.num_cities = num_cities

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> State:
        """Call method responsible for generating a new state.

        Args:
            key: jax random key in case stochasticity is used in the instance generation process.

        Returns:
            A `TSP` environment state.
        """


class UniformGenerator(Generator):
    """Instance generator that generates a random uniform instance of the traveling salesman
    problem. Given the number of cities, the coordinates of the cities are randomly sampled from a
    uniform distribution on the unit square.
    """

    def __init__(self, num_cities: int):
        super().__init__(num_cities)

    def __call__(self, key: chex.PRNGKey) -> State:
        key, sample_key = jax.random.split(key)

        # Randomly sample the coordinates of the cities.
        coordinates = jax.random.uniform(
            sample_key, (self.num_cities, 2), minval=0, maxval=1
        )

        # Initially, the position is set to -1, which means that the agent is not in any city.
        position = jnp.array(-1, jnp.int32)

        # Initially, the agent has not visited any city.
        visited_mask = jnp.zeros(self.num_cities, dtype=bool)
        trajectory = jnp.full(self.num_cities, -1, jnp.int32)

        # The number of visited cities is set to 0.
        num_visited = jnp.array(0, jnp.int32)

        state = State(
            coordinates=coordinates,
            position=position,
            visited_mask=visited_mask,
            trajectory=trajectory,
            num_visited=num_visited,
            key=key,
        )

        return state
