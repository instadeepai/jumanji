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

from typing import Callable, Optional, TypeVar

import chex
import jax

from jumanji import specs
from jumanji.env import Environment

Observation = TypeVar("Observation")
Action = TypeVar("Action")
SelectActionFn = Callable[[chex.PRNGKey, Observation], Action]


def make_random_select_action_fn(action_spec: specs.Spec) -> SelectActionFn:
    """Create select action function that chooses random actions."""

    def select_action(key: chex.PRNGKey, observation: chex.ArrayTree) -> chex.ArrayTree:
        if hasattr(observation, "action_mask"):
            return action_spec.sample(key, observation.action_mask)
        return action_spec.sample(key)

    return select_action


def check_env_does_not_smoke(
    env: Environment,
    select_action: Optional[SelectActionFn] = None,
    assert_finite_check: bool = True,
) -> None:
    """Run an episode of the environment, with a jitted step function to check no errors occur."""
    action_spec = env.action_spec()
    if select_action is None:
        select_action = make_random_select_action_fn(action_spec)

    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    state, timestep = env.reset(reset_key)
    step_fn = jax.jit(env.step)
    while not timestep.last():
        key, action_key = jax.random.split(key)
        action = select_action(action_key, timestep.observation)
        env.action_spec().validate(action)
        state, timestep = step_fn(state, action)
        env.observation_spec().validate(timestep.observation)
        if assert_finite_check:
            chex.assert_tree_all_finite((state, timestep))
