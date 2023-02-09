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

from typing import TYPE_CHECKING, NamedTuple

import chex
import jax.random

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

from chex import Array


@dataclass
class State:
    """
    weights: array of weights of the items.
    values: array of values of the items.
    packed_items: binary mask indicating if an item is in the knapsack (False/True <--> out/in).
    remaining_budget: the budget currently remaining.
    """

    weights: Array  # (num_items,)
    values: Array  # (num_items,)
    packed_items: Array  # (num_items,)
    remaining_budget: chex.Array  # (), jnp.float32
    key: chex.PRNGKey = jax.random.PRNGKey(0)


class Observation(NamedTuple):
    """
    weights: array of weights of the items.
    values: array of values of the items.
    action_mask: binary mask (False/True <--> illegal/legal).
    """

    weights: Array
    values: Array
    action_mask: Array
