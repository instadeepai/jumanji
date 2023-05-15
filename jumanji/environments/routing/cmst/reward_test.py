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

from jumanji.environments.routing.cmst.constants import (
    INVALID_ALREADY_TRAVERSED,
    INVALID_CHOICE,
    INVALID_NODE,
)
from jumanji.environments.routing.cmst.env import CMST
from jumanji.environments.routing.cmst.types import State
from jumanji.types import TimeStep


def test__cmst_dense_rewards(
    deterministic_cmst_env: Tuple[CMST, State, TimeStep]
) -> None:
    """Test that the defautl dense reward function works correctly."""

    # Default reward values are (0.1, -0.03, -0.01)
    # for (connection, no connection, invalid action)
    env, state, timestep = deterministic_cmst_env
    step_fn = jax.jit(env.step)

    action = jnp.array([4, 3])

    new_action, next_nodes = env._trim_duplicated_invalid_actions(
        state, action, state.key
    )

    assert new_action[1] == INVALID_CHOICE
    assert next_nodes[1] == INVALID_NODE

    state, timestep = step_fn(state, action)
    expected = jnp.sum(jnp.array([-0.03, -0.04]))

    assert jnp.array_equal(timestep.reward, expected)

    action = jnp.array([1, 7])
    new_action, next_nodes = env._trim_duplicated_invalid_actions(
        state, action, state.key
    )
    state, timestep = step_fn(state, action)

    assert new_action[0] == INVALID_ALREADY_TRAVERSED
    expected = jnp.sum(jnp.array([-0.03, -0.03]))
    assert jnp.array_equal(timestep.reward, expected)

    action = jnp.array([0, 8])
    state, timestep = step_fn(state, action)
    expected = jnp.sum(jnp.array([0.1, 0.1]))
    assert jnp.array_equal(timestep.reward, expected)
