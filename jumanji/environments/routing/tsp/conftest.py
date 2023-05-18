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
import jax.numpy as jnp
import pytest
import chex

from jumanji.environments.routing.tsp.env import TSP
from jumanji.environments.routing.tsp.generator import Generator, RandomGenerator
from jumanji.environments.routing.tsp.reward import DenseReward, SparseReward
from jumanji.environments.routing.tsp.types import State


@pytest.fixture
def dense_reward() -> DenseReward:
    return DenseReward()


@pytest.fixture
def sparse_reward() -> SparseReward:
    return SparseReward()


@pytest.fixture
def tsp_dense_reward(dense_reward: DenseReward) -> TSP:
    """Instantiates a TSP environment with dense rewards and 5 cities."""
    return TSP(generator=RandomGenerator(num_cities=5), reward_fn=dense_reward)


@pytest.fixture
def tsp_sparse_reward(sparse_reward: SparseReward) -> TSP:
    """Instantiates a TSP environment with sparse rewards and 5 cities."""
    return TSP(generator=RandomGenerator(num_cities=5), reward_fn=sparse_reward)


class DummyGenerator(Generator):
    """Hardcoded `Generator` mainly used for testing and debugging. It deterministically
    outputs a hardcoded instance with 5 cities.
    """

    def __init__(self) -> None:
        super().__init__(
            num_cities=5,
        )

    def __call__(self, key: PRNGKey) -> State:
        """Call method responsible for generating a new state. It returns a travelling salesman
        problem instance without any visited cities.

        Args:
            key: jax random key for any stochasticity used in the generation process. Not used
                in this instance generator.

        Returns:
            A TSP State.
        """
        del key

        coordinates = jnp.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]]
        )

        # Initially, the position is set to -1, which means that the agent is not in any city.
        position = jnp.array(-1, jnp.int32)

        # Initially, the agent has not visited any city.
        visited_mask = jnp.array([False, False, False, False, False])
        trajectory = jnp.array([-1, -1, -1, -1, -1], jnp.int32)

        # The number of visited cities is set to 0.
        num_visited = jnp.array(0, jnp.int32)

        state = State(
            coordinates=coordinates,
            position=position,
            visited_mask=visited_mask,
            trajectory=trajectory,
            num_visited=num_visited,
            key=jax.random.PRNGKey(0),
        )

        return state
