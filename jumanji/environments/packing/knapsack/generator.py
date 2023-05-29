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

from jumanji.environments.packing.knapsack.types import State


class Generator(abc.ABC):
    """Defines the abstract `Generator` base class. A `Generator` is responsible
    for generating a problem instance when the environment is reset.
    """

    def __init__(
        self,
        num_items: int,
        total_budget: float,
    ):
        """Abstract class implementing the attribute `num_cities`.

        Args:
            num_items (int): the number of items in the problem instance.
            total_budget (float): the total budget of the knapsack.
        """
        self.num_items = num_items
        self.total_budget = total_budget

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> State:
        """Call method responsible for generating a new state.

        Args:
            key: jax random key in case stochasticity is used in the instance generation process.

        Returns:
            A `Knapsack` environment state.
        """


class RandomGenerator(Generator):
    """Instance generator that generates an instance of the knapsack problem. Given the number of
    items and the maximum budget, the weights and values of the items are randomly sampled from a
    uniform distribution on the unit square.
    """

    def __init__(self, num_items: int, total_budget: float):
        super().__init__(num_items, total_budget)

    def __call__(self, key: chex.PRNGKey) -> State:
        key, sample_key = jax.random.split(key)

        # Sample weights and values of the items from a uniform distribution on [0, 1]
        weights, values = jax.random.uniform(
            sample_key, (2, self.num_items), minval=0, maxval=1
        )

        # Initially, no items are packed.
        packed_items = jnp.zeros(self.num_items, dtype=bool)

        # Initially, the remaining budget is the total budget.
        remaining_budget = jnp.array(self.total_budget, float)

        state = State(
            weights=weights,
            values=values,
            packed_items=packed_items,
            remaining_budget=remaining_budget,
            key=key,
        )

        return state
