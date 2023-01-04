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

from typing import Generic, NamedTuple, Optional

import chex
import haiku as hk
import jax.numpy as jnp
import optax

from jumanji.env import State
from jumanji.types import TimeStep


class Transition(NamedTuple):
    """Container for a transition."""

    observation: chex.ArrayTree
    action: chex.ArrayTree
    reward: chex.ArrayTree
    discount: chex.ArrayTree
    next_observation: chex.ArrayTree
    log_prob: chex.ArrayTree
    logits: chex.ArrayTree


class ActorCriticParams(NamedTuple):
    actor: hk.Params
    critic: hk.Params


class ParamsState(NamedTuple):
    """Container for the variables used during the training of an agent."""

    params: ActorCriticParams
    opt_state: optax.OptState
    update_count: jnp.float32  # use float to avoid int32 overflow


class ActingState(NamedTuple, Generic[State]):
    """Container for data used during the acting in the environment."""

    state: State
    timestep: TimeStep
    key: chex.PRNGKey
    episode_count: jnp.float32
    env_step_count: jnp.float32


class TrainingState(NamedTuple):
    """Container for data used during the training of an agent acting in an environment."""

    params_state: Optional[ParamsState]
    acting_state: ActingState
