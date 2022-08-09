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

from typing import NamedTuple, Optional

import haiku as hk
import jax.numpy as jnp
import optax
from chex import ArrayTree

from jumanji.types import Extra


class Transition(NamedTuple):
    """Container for a transition."""

    observation: ArrayTree
    action: ArrayTree
    reward: ArrayTree
    discount: ArrayTree
    next_observation: ArrayTree
    extra: Extra


class TrainingState(NamedTuple):
    """Container for the variables used during the training of an agent."""

    params: Optional[hk.Params] = None
    opt_state: Optional[optax.OptState] = None
    counter: jnp.int32 = jnp.int32(0)
