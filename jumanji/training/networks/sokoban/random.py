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

from jumanji.environments.routing.sokoban import Observation
from jumanji.training.networks.protocols import RandomPolicy


def categorical_random(
    observation: Observation,
    key: chex.PRNGKey,
) -> chex.Array:
    logits = jnp.zeros(shape=(observation.grid.shape[0], 4))

    action = jax.random.categorical(key, logits)
    return action


def make_random_policy_sokoban() -> RandomPolicy:
    """Make random policy for the `Sokoban` environment."""
    return categorical_random
