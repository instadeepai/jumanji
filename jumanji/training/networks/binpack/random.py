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

from jumanji.environments.packing.binpack import BinPack, Observation
from jumanji.training.networks.random_policy import RandomPolicy


def make_random_policy_binpack(binpack: BinPack) -> RandomPolicy:
    """Make random policy for BinPack."""
    num_ems, num_items = binpack.action_spec().num_values

    def random_policy(observation: Observation, key: chex.PRNGKey) -> chex.Array:
        action_mask = observation.action_mask.reshape(
            (*observation.action_mask.shape[:-2], -1)
        )
        flatten_logits = jnp.where(
            action_mask,
            jnp.zeros_like(action_mask),
            -jnp.finfo("float32").max,
        )
        flatten_action = jax.random.categorical(key, flatten_logits)
        ems_id, item_id = jnp.divmod(flatten_action, num_items)
        action = jnp.stack([ems_id, item_id], axis=-1)
        return action

    return random_policy
