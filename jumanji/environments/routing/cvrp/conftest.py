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

import chex
import jax
import jax.numpy as jnp
import pytest

from jumanji.environments.routing.cvrp.env import CVRP
from jumanji.environments.routing.cvrp.generator import (
    Generator,
    RandomUniformGenerator,
)
from jumanji.environments.routing.cvrp.reward import DenseReward, SparseReward
from jumanji.environments.routing.cvrp.types import State


@pytest.fixture
def dense_reward() -> DenseReward:
    return DenseReward()


@pytest.fixture
def sparse_reward() -> SparseReward:
    return SparseReward()


@pytest.fixture
def cvrp_dense_reward(dense_reward: DenseReward) -> CVRP:
    """Instantiates a CVRP environment with dense rewards and 5 nodes, maximum capacity of 3
    and maximum demand of 2.
    """
    return CVRP(
        generator=RandomUniformGenerator(num_nodes=5, max_capacity=3, max_demand=2),
        reward_fn=dense_reward,
    )


@pytest.fixture
def cvrp_sparse_reward(sparse_reward: SparseReward) -> CVRP:
    """Instantiates a CVRP environment with sparse rewards and 5 nodes, maximum capacity of 3
    and maximum demand of 2.
    """
    return CVRP(
        generator=RandomUniformGenerator(num_nodes=5, max_capacity=3, max_demand=2),
        reward_fn=sparse_reward,
    )


class DummyGenerator(Generator):
    """Hardcoded `Generator` mainly used for testing and debugging. It deterministically outputs a
    hardcoded instance with 4 nodes (cities), maximum vehcicle capacity of 6 and maximum city demand
    of 3.
    """

    def __init__(self) -> None:
        super().__init__(num_nodes=4, max_capacity=6, max_demand=3)

    def __call__(self, key: chex.PRNGKey) -> State:
        """Call method responsible for generating a new state. It returns a capacitated vehicle
        routing problem without any visited cities and starting at the depot node.

        Args:
            key: jax random key for any stochasticity used in the generation process. Not used
                in this instance generator.
        Returns:
            A CVRP State.
        """
        del key

        coordinates = jnp.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]], float
        )
        demands = jnp.array([0, 1, 2, 1, 2], jnp.int32)

        # The initial position is set at the depot.
        position = jnp.array(0, jnp.int32)
        num_total_visits = jnp.array(1, jnp.int32)

        # The initial capacity is set to the maximum capacity.
        capacity = jnp.array(6, jnp.int32)

        # Initially, the agent has only visited the depot.
        visited_mask = jnp.array([True, False, False, False, False], bool)
        trajectory = jnp.array([0, 0, 0, 0, 0, 0, 0, 0], jnp.int32)

        state = State(
            coordinates=coordinates,
            demands=demands,
            position=position,
            capacity=capacity,
            visited_mask=visited_mask,
            trajectory=trajectory,
            num_total_visits=num_total_visits,
            key=jax.random.PRNGKey(0),
        )

        return state
