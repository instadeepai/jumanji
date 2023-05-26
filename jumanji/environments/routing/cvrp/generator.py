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

from jumanji.environments.routing.cvrp.constants import DEPOT_IDX
from jumanji.environments.routing.cvrp.types import State


class Generator(abc.ABC):
    """Defines the abstract `Generator` base class. A `Generator` is responsible
    for generating a problem instance when the environment is reset.
    """

    def __init__(
        self,
        num_nodes: int,
        max_capacity: int,
        max_demand: int,
    ):
        """Abstract class implementing the attributes `num_nodes`, `max_capacity`, `max_demand`.

        Args:
            num_nodes (int): the number of cities in the problem instance.
            max_capacity (int): the maximum capacity of the vehicles.
            max_demand (int): the maximum demand of the nodes.
        """
        self.num_nodes = num_nodes
        self.max_capacity = max_capacity
        self.max_demand = max_demand

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> State:
        """Call method responsible for generating a new state.

        Args:
            key: jax random key in case stochasticity is used in the instance generation process.

        Returns:
            A `CVRP` environment state.
        """


class UniformGenerator(Generator):
    """Instance generator that generates a random uniform instance of the capacitated vehicle
    routing problem. Given the number of nodes, maximum capacity of the vehicle and maximum demand
    of the nodes, the generation works as follows. The coordinates of the cities and the depot are
    randomly sampled from a uniform distribution on the unit square. The demands of the nodes are
    randomly sampled integers from the interval [1, max_demand], and the demand of the depot is set
    to 0.
    """

    def __init__(self, num_nodes: int, max_capacity: int, max_demand: int):
        """Instantiates a `UniformGenerator`.

        Args:
            num_nodes: number of city nodes in the environment.
            max_capacity: maximum capacity of the vehicle.
            max_demand: maximum demand of each node.
        """
        super().__init__(num_nodes, max_capacity, max_demand)

    def __call__(self, key: chex.PRNGKey) -> State:
        key, coordinates_key, demands_key = jax.random.split(key, 3)

        # Randomly sample the coordinates of the cities.
        coordinates = jax.random.uniform(
            coordinates_key, (self.num_nodes + 1, 2), minval=0, maxval=1
        )

        # Randomly sample the demands of the nodes.
        demands = jax.random.randint(
            demands_key, (self.num_nodes + 1,), minval=1, maxval=self.max_demand
        )

        # Set the depot demand to 0.
        demands = demands.at[DEPOT_IDX].set(0)

        # The initial position is set at the depot.
        position = jnp.array(DEPOT_IDX, jnp.int32)
        num_total_visits = jnp.array(1, jnp.int32)

        # The initial capacity is set to the maximum capacity.
        capacity = jnp.array(self.max_capacity, jnp.int32)

        # Initially, the agent has only visited the depot.
        visited_mask = jnp.zeros(self.num_nodes + 1, dtype=bool).at[DEPOT_IDX].set(True)
        trajectory = jnp.full(2 * self.num_nodes, DEPOT_IDX, jnp.int32)

        state = State(
            coordinates=coordinates,
            demands=demands,
            position=position,
            capacity=capacity,
            visited_mask=visited_mask,
            trajectory=trajectory,
            num_total_visits=num_total_visits,
            key=key,
        )

        return state
