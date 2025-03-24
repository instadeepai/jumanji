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

import jax
import numpy as np

from jumanji.environments.routing.multi_cvrp.constants import DEPOT_IDX
from jumanji.environments.routing.multi_cvrp.utils import (
    compute_distance,
    compute_time_penalties,
    generate_uniform_random_problem,
    get_init_settings,
)


class TestObservationSpec:
    def test_compute_time_penalties(self) -> None:
        """Test whether the compute_time_pentalties function works correctly."""
        local_times = np.array([1.0, 2.5, 4.0])
        window_start = np.array([2.0, 2.0, 2.0])
        window_end = np.array([3.0, 3.0, 3.0])
        early_coefs = np.array([0.1, 0.15, 0.09])
        late_coefs = np.array([0.5, 0.3, 0.7])

        pentalties = compute_time_penalties(
            local_times, window_start, window_end, early_coefs, late_coefs
        )

        assert np.array_equal(pentalties, np.array([0.1, 0.0, 0.7], dtype=np.float32))

    def test_compute_distance(self) -> None:
        """Test whether the compute_time_pentalties function works correctly."""
        start_node_coordinates = np.array([[1.0, 1.0], [1.0, 2.5]])
        end_node_coordinates = np.array([[2.0, 1.0], [1.0, 0.5]])

        distances = compute_distance(start_node_coordinates, end_node_coordinates)

        assert np.array_equal(distances, np.array([1.0, 2.0], dtype=np.float32))

    def test_generate_uniform_random_problem(self) -> None:
        """Test whether a problem can be generated correctly using the 150 customer setting."""

        key = jax.random.PRNGKey(1234)
        window_length = 20

        num_vehicles = 5
        num_customers = 150
        (
            map_max,
            max_capacity,
            max_start_window,
            early_coef_rand,
            late_coef_rand,
            customer_demand_max,
        ) = get_init_settings(num_customers, num_vehicles)
        total_capacity = max_capacity * num_vehicles

        (
            node_coordinates,
            node_demands,
            window_start_times,
            window_end_times,
            early_coefs,
            late_coefs,
        ) = generate_uniform_random_problem(
            key,
            num_customers,
            total_capacity,
            map_max,
            customer_demand_max,
            max_start_window,
            window_length,
            early_coef_rand,
            late_coef_rand,
        )

        assert node_coordinates.shape == (num_customers + 1, 2)
        assert np.max(node_coordinates) < map_max
        assert np.min(node_coordinates) >= 0

        assert node_demands.shape == (num_customers + 1,)
        assert np.min(node_demands) >= 0
        assert node_demands[DEPOT_IDX] == 0

        assert window_start_times.shape == (num_customers + 1,)
        assert np.min(window_start_times) >= 0
        assert np.min(window_start_times) < max_start_window

        assert np.array_equal(window_start_times + window_length, window_end_times)

        assert np.array_equal(
            early_coefs, np.array([0.0] + [0.1] * num_customers, dtype=np.float32)
        )
        assert np.array_equal(late_coefs, np.array([0.0] + [0.5] * num_customers, dtype=np.float32))
