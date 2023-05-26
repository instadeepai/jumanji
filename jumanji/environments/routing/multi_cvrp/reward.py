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

from jumanji.environments.routing.multi_cvrp.types import State
from jumanji.environments.routing.multi_cvrp.utils import max_single_vehicle_distance


class RewardFn(abc.ABC):
    def __init__(self, num_vechicles: int, num_customers: int, map_max: int) -> None:
        self._num_vehicles = num_vechicles
        self._num_customers = num_customers
        self._map_max = map_max
        # This is the maximum negative reward that can be given to an agent.
        self._large_negate_reward = (
            -max_single_vehicle_distance(self._map_max, self._num_customers)
            * self._num_vehicles
        )

    @abc.abstractmethod
    def __call__(
        self,
        state: State,
        is_done: bool,
    ) -> chex.Numeric:
        """Compute the reward based on the current state, the chosen action, the next state and
        whether the action is valid.
        """


class SparseReward(RewardFn):
    """The negative distance between the current city and the chosen next city to go to length at
    the end of the episode. It also includes the distance to the depot to complete the tour.
    Note that the reward is `-2 * num_nodes * sqrt(2)` if the chosen action is invalid.
    """

    def __call__(
        self,
        state: State,
        is_done: bool,
    ) -> chex.Numeric:
        def compute_episode_reward(state: State) -> float:
            return jax.lax.cond(  # type: ignore
                jnp.any(state.step_count > self._num_customers * 2),
                # Penalise for running into step limit. This is not including max time
                # penalties as the distance penalties are already enough.
                lambda state: self._large_negate_reward,
                lambda state: -state.vehicles.distances.sum()
                - state.vehicles.time_penalties.sum(),
                state,
            )

        # By default, returns the negative distance between the previous and new node.
        reward = jax.lax.select(
            is_done,
            compute_episode_reward(state),
            jnp.float32(0),
        )

        return reward
