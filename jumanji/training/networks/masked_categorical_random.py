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
from typing_extensions import Protocol


class ObservationWithActionMaskProtocol(Protocol):
    """Make sure that the Observation has a field called action_mask."""

    action_mask: chex.Array


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
