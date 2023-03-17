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


class RewardFn(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        state: State,
        action: chex.Numeric,
        next_state: State,
        is_valid: bool,
    ) -> chex.Numeric:
        """Compute the reward based on the current state, the chosen action, the next state
        and whether the action is valid.
        """


class SparseReward(RewardFn):
    """The negative tour length at the end of the episode. The tour length is defined as the sum
    of the distances between consecutive cities. It is computed by starting at the first city
    and ending there, after visiting all the cities.
    Note that the reward is 0 unless the episode terminates, it is `-num_cities * sqrt(2)`
    if the chosen action is invalid.
    """

    def __call__(
        self,
        state: State,
        action: chex.Array,
        next_state: State,
        is_valid: bool,
    ) -> chex.Numeric:
        num_cities = len(state.visited_mask)
        compute_sparse_reward = lambda: jax.lax.select(
            is_valid,
            -compute_tour_length(next_state.coordinates, next_state.trajectory),
            jnp.array(-num_cities * jnp.sqrt(2), float),
        )
        is_done = (next_state.num_visited == num_cities) | ~is_valid
        reward = jax.lax.cond(
            is_done,
            compute_sparse_reward,
            lambda: jnp.array(0, float),
        )
        return reward


class DenseReward(RewardFn):
    """The negative distance between the current city and the chosen next city to go to.
    It is 0 for the first chosen city, and for the last city, it also includes the distance
    to the initial city to complete the tour.
    Note that the reward is `-num_cities * sqrt(2)` if the chosen action is invalid.
    """

    def __call__(
        self, state: State, action: chex.Array, next_state: State, is_valid: bool
    ) -> chex.Numeric:
        num_cities = len(state.visited_mask)
        previous_city = state.coordinates[state.position]
        next_city = next_state.coordinates[next_state.position]
        # By default, returns the negative distance between the previous and new city.
        reward = jax.lax.select(
            is_valid,
            -distance_between_two_cities(previous_city, next_city),
            jnp.array(-num_cities * jnp.sqrt(2), float),
        )
        # Returns 0 for the first city selected.
        reward = jax.lax.select(
            state.num_visited == 0,
            jnp.array(0, float),
            reward,
        )
        # Adds the distance between the last city and the first city if the tour is finished.
        initial_city = state.coordinates[state.trajectory[0]]
        reward = jax.lax.select(
            jnp.all(next_state.visited_mask),
            reward - distance_between_two_cities(next_city, initial_city),
            reward,
        )
        return reward


def compute_tour_length(
    coordinates: chex.Array, trajectory: chex.Array
) -> chex.Numeric:
    sorted_coordinates = coordinates[trajectory]
    # Shift coordinates to compute the distance between neighboring cities.
    shifted_coordinates = jnp.roll(sorted_coordinates, -1, axis=0)
    return jnp.linalg.norm((sorted_coordinates - shifted_coordinates), axis=1).sum()


def distance_between_two_cities(
    city_one_coordinates: chex.Array, city_two_coordinates: chex.Array
) -> chex.Numeric:
    """Calculate the Euclidean distance between two neighboring cities."""
    return jnp.linalg.norm(city_one_coordinates - city_two_coordinates)
