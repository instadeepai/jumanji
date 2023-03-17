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

import pytest

from jumanji.environments.packing.knapsack.env import Knapsack
from jumanji.environments.packing.knapsack.reward import DenseReward, SparseReward


@pytest.fixture
def dense_reward() -> DenseReward:
    return DenseReward()


@pytest.fixture
def sparse_reward() -> SparseReward:
    return SparseReward()


@pytest.fixture
def knapsack_dense_reward(dense_reward: DenseReward) -> Knapsack:
    """Instantiates a Knapsack environment with dense rewards, 10 items and a budget of 2.0."""
    return Knapsack(num_items=10, total_budget=2.0, reward_fn=dense_reward)


@pytest.fixture
def knapsack_sparse_reward(sparse_reward: SparseReward) -> Knapsack:
    """Instantiates a Knapsack environment with sparse rewards, 10 items and a budget of 2.0."""
    return Knapsack(num_items=10, total_budget=2.0, reward_fn=sparse_reward)
