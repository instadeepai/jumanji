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

from typing import Callable, Optional, TypeVar, Union

import chex
import jax
import jax.numpy as jnp

from jumanji import specs
from jumanji.env import Environment

Observation = TypeVar("Observation")
Action = TypeVar("Action")
SelectActionFn = Callable[[chex.PRNGKey, Observation], Action]


def make_random_select_action_fn(
    action_spec: Union[
        specs.BoundedArray, specs.DiscreteArray, specs.MultiDiscreteArray
    ]
) -> SelectActionFn:
    """Create select action function that chooses random actions."""

    def select_action(key: chex.PRNGKey, state: chex.ArrayTree) -> chex.ArrayTree:
        del state
        if (
            isinstance(action_spec, specs.DiscreteArray)
            or isinstance(action_spec, specs.MultiDiscreteArray)
            or jnp.issubdtype(action_spec.dtype, jnp.integer)
        ):
            action = jax.random.randint(
                key=key,
                shape=action_spec.shape,
                minval=action_spec.minimum,
                maxval=action_spec.maximum + 1,
                dtype=action_spec.dtype,
            )
        elif isinstance(action_spec, specs.BoundedArray):
            assert jnp.issubdtype(action_spec.dtype, jnp.floating)
            action = jax.random.uniform(
                key=key,
                shape=action_spec.shape,
                dtype=action_spec.dtype,
                minval=action_spec.minimum,
                maxval=action_spec.maximum,
            )
        else:
            raise ValueError(
                "Only supported for action specs of type `specs.BoundedArray, "
                "specs.DiscreteArray or specs.MultiDiscreteArray`."
            )
        return action

    return select_action


def check_env_does_not_smoke(
    env: Environment,
    select_action: Optional[SelectActionFn] = None,
    assert_finite_check: bool = True,
) -> None:
    """Run an episode of the environment, with a jitted step function to check no errors occur."""
    action_spec = env.action_spec()
    if select_action is None:
        if isinstance(action_spec, specs.BoundedArray) or isinstance(
            action_spec, specs.DiscreteArray
        ):
            select_action = make_random_select_action_fn(action_spec)
        else:
            raise NotImplementedError(
                f"Currently the `make_random_select_action_fn` only works for environments with "
                f"either discrete actions or bounded continuous actions. The input environment to "
                f"this test has an action spec of type {action_spec}, and therefore requires "
                f"a custom `SelectActionFn` to be provided to this test."
            )
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
