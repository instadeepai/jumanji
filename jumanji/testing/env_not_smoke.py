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
import numpy as np

from jumanji import Environment, specs

Observation = TypeVar("Observation")
Action = TypeVar("Action")
SelectActionFn = Callable[[chex.PRNGKey, Observation], Action]


def make_random_select_action_fn(
    action_spec: Union[specs.BoundedArray, specs.DiscreteArray]
) -> SelectActionFn:
    """Create select action function that chooses random actions."""

    def select_action(key: chex.PRNGKey, state: chex.ArrayTree) -> chex.ArrayTree:
        del state
        # TODO: Change to Discrete or Continuous Array Spec condition once MultipleDiscrete is
        #  handled.
        if (
            np.issubdtype(action_spec.dtype, np.int32)
            or np.issubdtype(action_spec.dtype, np.int64)
            or np.issubdtype(action_spec.dtype, np.int16)
            or np.issubdtype(action_spec.dtype, np.int8)
        ):
            action = jax.random.randint(
                key=key,
                shape=action_spec.shape,
                minval=action_spec.minimum,
                maxval=action_spec.maximum + 1,
                dtype=action_spec.dtype,
            )
        elif (
            np.issubdtype(action_spec.dtype, np.float32)
            or np.issubdtype(action_spec.dtype, np.float64)
            or np.issubdtype(action_spec.dtype, np.float16)
            or np.issubdtype(action_spec.dtype, np.float8)
        ):
            action = jax.random.uniform(
                key=key,
                shape=action_spec.shape,
                dtype=action_spec.dtype,
                minval=action_spec.minimum,
                maxval=action_spec.maximum,
            )
        else:
            raise ValueError(
                f"`action_spec.dtype` must be integer or float, got {action_spec.dtype}."
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
    if not select_action:
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
    key, subkey = jax.random.split(key)
    state, timestep, extra = env.reset(subkey)
    step_fn = jax.jit(env.step)
    while not timestep.last():
        key, subkey = jax.random.split(key)
        action = select_action(subkey, timestep.observation)

        state, timestep, extra = step_fn(state, action)
        # TODO: uncomment below lines after the 32 bit/64 bit issue
        #  (https://gitlab.com/instadeep/jumanji/-/issues/84) is resolved.
        # env.action_spec().validate(action)
        # env.observation_spec().validate(timestep.observation)
        if assert_finite_check:
            chex.assert_tree_all_finite((state, timestep))
