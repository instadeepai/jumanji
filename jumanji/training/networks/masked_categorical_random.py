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

from typing import TypeVar

import chex
import jax
import jax.numpy as jnp

from jumanji.training.networks.protocols import (
    ObservationWithActionMaskProtocol,
    RandomPolicy,
)

ObservationWithActionMask = TypeVar(
    "ObservationWithActionMask", bound="ObservationWithActionMaskProtocol"
)


def masked_categorical_random(
    observation: ObservationWithActionMask,
    key: chex.PRNGKey,
) -> chex.Array:
    logits = jnp.where(
        observation.action_mask,
        jnp.zeros_like(observation.action_mask),
        -jnp.finfo("float32").max,
    )
    action = jax.random.categorical(key, logits)
    return action


def make_masked_categorical_random_ndim(
    action_spec_num_values: chex.Array,
) -> RandomPolicy:
    def policy(observation: ObservationWithActionMask, key: chex.PRNGKey) -> chex.Array:
        """Sample uniformly at random from a joint distribution with masking"""
        n = action_spec_num_values.shape[0]
        action_mask = observation.action_mask.reshape(
            (observation.action_mask.shape[0], -1)
        )
        flatten_logits = jnp.where(
            action_mask,
            jnp.zeros_like(action_mask),
            -jnp.finfo("float32").max,
        )
        flat_action = jax.random.categorical(key, flatten_logits)
        action_components = []
        for i in range(n - 1, 0, -1):
            flat_action, remainder = jnp.divmod(flat_action, action_spec_num_values[i])
            action_components.append(remainder)
        action_components.append(flat_action)
        action = jnp.stack(list(reversed(action_components)), axis=-1)
        return action

    return policy
