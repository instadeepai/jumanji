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
import jax.random

from jumanji.environments.swarms.search_and_rescue import SearchAndRescue
from jumanji.environments.swarms.search_and_rescue.types import Observation
from jumanji.training.networks.protocols import RandomPolicy


class SearchAndRescueRandomPolicy(RandomPolicy):
    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def __call__(
        self,
        observation: Observation,
        key: chex.PRNGKey,
    ) -> chex.Array:
        """A random policy given an environment-specific observation.

        Args:
            observation: environment observation.
            key: random key for action selection.

        Returns:
            action
        """
        shape = (
            observation.searcher_views.shape[0],
            observation.searcher_views.shape[1],
            self.n_actions,
        )
        return jax.random.uniform(key, shape, minval=-1.0, maxval=1.0)


def make_random_policy_search_and_rescue(search_and_rescue: SearchAndRescue) -> RandomPolicy:
    """Make random policy for Search & Rescue."""
    return SearchAndRescueRandomPolicy(search_and_rescue.action_spec.shape[-1])
