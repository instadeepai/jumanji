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

from jumanji.environments.packing.knapsack.env import Knapsack
from jumanji.environments.packing.knapsack.generator import Generator, RandomGenerator
from jumanji.environments.packing.knapsack.reward import DenseReward, SparseReward
from jumanji.environments.packing.knapsack.types import State


@pytest.fixture
def dense_reward() -> DenseReward:
    return DenseReward()


@pytest.fixture
def sparse_reward() -> SparseReward:
    return SparseReward()


@pytest.fixture
def knapsack_dense_reward(dense_reward: DenseReward) -> Knapsack:
    """Instantiates a Knapsack environment with dense rewards, 10 items and a budget of 2.0."""
    return Knapsack(
        generator=RandomGenerator(num_items=10, total_budget=2.0),
        reward_fn=dense_reward,
    )


@pytest.fixture
def knapsack_sparse_reward(sparse_reward: SparseReward) -> Knapsack:
    """Instantiates a Knapsack environment with sparse rewards, 10 items and a budget of 2.0."""
    return Knapsack(
        generator=RandomGenerator(num_items=10, total_budget=2.0),
        reward_fn=sparse_reward,
    )


class DummyGenerator(Generator):
    """Hardcoded `Generator` mainly used for testing and debugging. it deterministically outputs a
    hardcoded instance with 5 items and a total budget of 2.5.
    """

    def __init__(self) -> None:
        super().__init__(num_items=5, total_budget=2.5)

    def __call__(self, key: chex.PRNGKey) -> State:
        """Call method responsible for generating a new state. It returns a knapsack problem
        instance without any items placed.

        Args:
            key: jax random key for any stochasticity used in the generation process. Not used
                in this instance generator.
        Returns:
            A Knapsack State.
        """
        del key

        weights = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9], float)
        values = jnp.array([0.3, 0.4, 0.5, 0.6, 0.7], float)

        # Initially, no items are packed.
        packed_items = jnp.array([False, False, False, False, False], dtype=bool)

        # Initially, the remaining budget is the total budget.
        remaining_budget = jnp.array(2.5, float)

        state = State(
            weights=weights,
            values=values,
            packed_items=packed_items,
            remaining_budget=remaining_budget,
            key=jax.random.PRNGKey(0),
        )

        return state
