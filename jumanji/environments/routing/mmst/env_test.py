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

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.mmst.constants import (
    INVALID_CHOICE,
    INVALID_TIE_BREAK,
)
from jumanji.environments.routing.mmst.env import MMST
from jumanji.environments.routing.mmst.types import State
from jumanji.testing.env_not_smoke import (
    check_env_does_not_smoke,
    check_env_specs_does_not_smoke,
)
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import TimeStep


def test__mmst_agent_observation(
    deterministic_mmst_env: Tuple[MMST, State, TimeStep],
) -> None:
    """Test that agent observation view of the node types is correct"""

    _, _, timestep = deterministic_mmst_env

    # nodes =  0, 1, 2, 3,  4, 5, 6, 7, 8, 9 10  11.
    # node_types = jnp.array([0, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, -1], dtype=jnp.int32).
    # conn_nodes = jnp.array([[1, -1, -1], [3, -1, -1]], dtype=jnp.int32).

    obs_node_types = jnp.array(
        [1, 0, -1, 2, -1, 3, 1, -1, 3, 1, -1, -1],
        dtype=jnp.int32,
    )

    assert jnp.array_equal(timestep.observation.node_types, obs_node_types)


def test__mmst_action_tie_break(
    deterministic_mmst_env: Tuple[MMST, State, TimeStep],
) -> None:
    """Test if the actions are mask correctly if multiple agents select the same node
    as next nodes.
    """

    env, state, _ = deterministic_mmst_env
    key1, key2 = jax.random.split(state.key)

    action = jnp.array([4, 4], dtype=jnp.int32)
    new_action, _ = env._trim_duplicated_invalid_actions(state, action, key1)
    assert not jnp.array_equal(action, new_action)
    assert INVALID_TIE_BREAK in new_action

    action2 = jnp.array([4, 7], dtype=jnp.int32)
    new_action2, _ = env._trim_duplicated_invalid_actions(state, action2, key2)
    assert jnp.array_equal(action2, new_action2)


def test__mmst_split_gn_reset(
    mmst_split_gn_env: MMST,
) -> None:
    """Validates the jitted reset of the environment."""
    chex.clear_trace_counter()
    reset_fn = jax.jit(chex.assert_max_traces(mmst_split_gn_env.reset, n=1))

    key = jax.random.PRNGKey(0)
    state, timestep = reset_fn(key)
    # Call again to check it does not compile twice.
    state, timestep = reset_fn(key)

    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)
    assert state.step_count == 0

    # Assert initial position is equal to current node.
    assert jnp.all(state.connected_nodes[:, 0] == state.positions)
    # Assert no agent is done.
    assert jnp.all(~state.finished_agents)


def test__mmst_step(mmst_split_gn_env: MMST) -> None:
    """Validates the jitted step of the environment."""
    chex.clear_trace_counter()

    step_fn = chex.assert_max_traces(mmst_split_gn_env.step, n=1)
    step_fn = jax.jit(step_fn)

    key = jax.random.PRNGKey(0)
    state, timestep = mmst_split_gn_env.reset(key)

    logits = jnp.where(
        state.action_mask,
        jnp.zeros_like(state.action_mask),
        jnp.finfo("float32").min,
    )

    action_key = jax.random.PRNGKey(1)
    action = jax.random.categorical(action_key, logits)
    new_state, next_timestep = step_fn(state, action)

    # Check that the state has changed.
    assert not jnp.array_equal(new_state.connected_nodes, state.connected_nodes)
    assert not jnp.array_equal(new_state.position_index, state.position_index)
    assert not jnp.array_equal(new_state.positions, state.positions)
    assert not jnp.array_equal(new_state.node_edges, state.node_edges)
    assert not jnp.array_equal(new_state.action_mask, state.action_mask)
    assert not jnp.array_equal(new_state.step_count, state.step_count)
    assert not jnp.array_equal(next_timestep.reward, timestep.reward)

    # Check that the state is made of DeviceArrays, this is false for the non-jitted.
    # step function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(new_state)


def test__mmst_does_not_smoke(
    mmst_split_gn_env: MMST,
) -> None:
    """Test that we can run an episode without any errors."""
    check_env_does_not_smoke(mmst_split_gn_env)


def test__mmst_specs_does_not_smoke(mmst_split_gn_env: MMST) -> None:
    """Test that we can access specs without any errors."""
    check_env_specs_does_not_smoke(mmst_split_gn_env)


def test__mmst_termination(
    deterministic_mmst_env: Tuple[MMST, State, TimeStep],
) -> None:
    env, state, timestep = deterministic_mmst_env
    step_fn = jax.jit(env.step)

    action = jnp.array([0, 4])
    state, timestep = step_fn(state, action)

    action = jnp.array([3, 5])
    state, timestep = step_fn(state, action)

    action = jnp.array([7, 4])
    state, timestep = step_fn(state, action)

    action = jnp.array([6, 8])
    state, timestep = step_fn(state, action)

    assert state.finished_agents[1]

    action = jnp.array([10, 10])
    _, step_key = jax.random.split(state.key)
    new_action, _ = env._trim_duplicated_invalid_actions(state, action, step_key)
    assert new_action[1] == INVALID_CHOICE

    state, timestep = step_fn(state, action)

    action = jnp.array([9, 4])
    state, timestep = step_fn(state, action)

    assert jnp.all(state.finished_agents)

    # Termination.
    assert timestep.last()
    assert jnp.all(timestep.discount == 0)


def test__mmst_truncation(deterministic_mmst_env: Tuple[MMST, State, TimeStep]) -> None:
    env, state, timestep = deterministic_mmst_env
    step_fn = jax.jit(env.step)

    # Truncation.
    for _ in range(env.time_limit + 1):
        state, timestep = step_fn(state, jnp.array([3, 3]))

    assert timestep.last()


def test__mmst_action_masking(
    deterministic_mmst_env: Tuple[MMST, State, TimeStep],
) -> None:
    env, state, _ = deterministic_mmst_env
    step_fn = jax.jit(env.step)

    assert state.action_mask[1, 4]

    action = jnp.array([4, 3])
    new_state, _ = step_fn(state, action)

    # Agent 1 shouldn't be able to acess node 4 any more.
    assert jnp.array_equal(state.positions[1], new_state.positions[1])
    assert not jnp.array_equal(state.action_mask[1], new_state.action_mask[1])
