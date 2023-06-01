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
from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.multi_cvrp.types import (
    Node,
    PenalityCoeff,
    State,
    StateVehicle,
    TimeWindow,
)
from jumanji.environments.routing.multi_cvrp.utils import (
    DEPOT_IDX,
    create_action_mask,
    generate_uniform_random_problem,
    get_init_settings,
)


class Generator(abc.ABC):
    """Base class for generators for the MACVPR environment."""

    def __init__(
        self,
        num_customers: int,
        num_vehicles: int,
        manual_settings: bool = False,
        map_max: int = 10,
        max_capacity: int = 60,
        max_start_window: float = 10.0,
        early_coef_rand: Tuple[float, float] = (0.0, 0.2),
        late_coef_rand: Tuple[float, float] = (0.0, 1.0),
        customer_demand_max: int = 10,
    ) -> None:
        """Initialises a MultiCVRP generator, used to generate the problem.

        Args:
            num_customers: number of customer nodes in the environment.
            num_vehicles: number of vehicles in the environment.
            manual_settings: whether to use the manual settings or the predefine settings
                from the paper.
            map_max: maximum value for the map size.
            max_capacity: maximum value for the vehicle capacity.
            max_start_window: maximum value for the start window.
            early_coef_rand: range for the early coefficient.
            late_coef_rand: range for the late coefficient.
            customer_demand_max: maximum value for the customer demand.
        """
        self._num_customers = num_customers
        self._num_vehicles = num_vehicles

        if not manual_settings:
            # Scenario are taken from the paper
            # Note: The time window detail could not be found in the paper for
            # the 20, 50 and 150 customer scenarios. We use the 150 customer scenario's
            # time window of 20 for them.
            self._time_window_length = 20
            (
                self._map_max,
                self._max_capacity,
                self._max_start_window,
                self._early_coef_rand,
                self._late_coef_rand,
                self._customer_demand_max,
            ) = get_init_settings(self.num_customers, self.num_vehicles)

        else:
            self._map_max = map_max
            self._max_capacity = max_capacity
            self._max_start_window = max_start_window
            self._early_coef_rand = early_coef_rand
            self._late_coef_rand = late_coef_rand
            self._customer_demand_max = customer_demand_max

        self._max_end_window = self._max_start_window + self._time_window_length

    @property
    def num_customers(self) -> int:
        return self._num_customers

    @property
    def num_vehicles(self) -> int:
        return self._num_vehicles

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `MultiCVRP` state.

        Returns:
            A `MultiCVRP` state.
        """


class UniformRandomGenerator(Generator):
    """Randomly (uniformly) places the customers and vehicles on a fixed size map."""

    def __init__(self, num_customers: int, num_vehicles: int) -> None:
        """Initialises a MultiCVRP generator, used to generate the problem.

        Args:
            num_customers: number of customer nodes in the environment.
            num_vehicles: number of vehicles in the environment.
        """
        super().__init__(num_customers, num_vehicles)

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `MultiCVRP` state.

        Returns:
            A `MultiCVRP` state.
        """

        # This split is uncessary, but it makes the code more readable.
        problem_key, key = jax.random.split(key)

        total_capacity = self._max_capacity * self._num_vehicles

        (
            node_coordinates,
            node_demands,
            window_start_times,
            window_end_times,
            early_coefs,
            late_coefs,
        ) = generate_uniform_random_problem(
            problem_key,
            self._num_customers,
            total_capacity,
            self._map_max,
            self._customer_demand_max,
            self._max_start_window,
            self._time_window_length,
            self._early_coef_rand,
            self._late_coef_rand,
        )
        capacities = jnp.ones(self._num_vehicles, dtype=jnp.int16) * self._max_capacity
        state = State(
            nodes=Node(coordinates=node_coordinates, demands=node_demands),
            windows=TimeWindow(start=window_start_times, end=window_end_times),
            coeffs=PenalityCoeff(early=early_coefs, late=late_coefs),
            vehicles=StateVehicle(
                positions=jnp.int16([DEPOT_IDX] * self._num_vehicles),
                local_times=jnp.zeros(self._num_vehicles, dtype=jnp.float32),
                capacities=capacities,
                distances=jnp.zeros(self._num_vehicles, dtype=jnp.float32),
                time_penalties=jnp.zeros(self._num_vehicles, dtype=jnp.float32),
            ),
            order=jnp.zeros(
                (self._num_vehicles, 2 * self._num_customers), dtype=jnp.int16
            ),
            action_mask=create_action_mask(node_demands, capacities),
            step_count=jnp.ones((), dtype=jnp.int16),
            key=key,
        )

        return state
