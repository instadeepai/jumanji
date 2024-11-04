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

from typing import Tuple

import jax
import jax.numpy as jnp

from jumanji.environments.routing.mmst.constants import (
    INVALID_ALREADY_TRAVERSED,
    INVALID_CHOICE,
    INVALID_NODE,
)
from jumanji.environments.routing.mmst.env import MMST
from jumanji.environments.routing.mmst.types import State
from jumanji.types import TimeStep


def test__mmst_dense_rewards(deterministic_mmst_env: Tuple[MMST, State, TimeStep]) -> None:
    """Test that the default dense reward function works correctly."""

    # Default reward values are (10.0, -1.0, -1.0)
    # for (connection, no connection, invalid action) repestitively.
    env, state, timestep = deterministic_mmst_env
    step_fn = jax.jit(env.step)

    action = jnp.array([4, 3])

    new_action, next_nodes = env._trim_duplicated_invalid_actions(state, action, state.key)

    assert new_action[1] == INVALID_CHOICE
    assert next_nodes[1] == INVALID_NODE

    state, timestep = step_fn(state, action)
    expected = jnp.sum(jnp.array([-1.0, -2.0]))

    assert jnp.array_equal(timestep.reward, expected)

    action = jnp.array([1, 7])
    new_action, next_nodes = env._trim_duplicated_invalid_actions(state, action, state.key)
    state, timestep = step_fn(state, action)

    assert new_action[0] == INVALID_ALREADY_TRAVERSED
    expected = jnp.sum(jnp.array([-1.0, -1.0]))
    assert jnp.array_equal(timestep.reward, expected)

    action = jnp.array([0, 8])
    state, timestep = step_fn(state, action)
    expected = jnp.sum(jnp.array([10.0, 10.0]))
    assert jnp.array_equal(timestep.reward, expected)
