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
import pytest
from jax import numpy as jnp
from jax import random

from jumanji.environments.packing.knapsack import Knapsack, State
from jumanji.testing.env_not_smoke import check_env_does_not_smoke
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import StepType, TimeStep


@pytest.fixture
def knapsack_env() -> Knapsack:
    """Instantiates a default knapsack environment."""
    return Knapsack()


def test_knapsack__reset(knapsack_env: Knapsack) -> None:
    """Validates the jitted reset of the environment."""

    reset_fn = jax.jit(knapsack_env.reset)
    key = random.PRNGKey(0)
    state, timestep = reset_fn(key)
    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)
    assert state.remaining_budget == knapsack_env.total_budget
    assert state.remaining_budget > 0
    assert jnp.all(state.packed_items == 0)

    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # reset function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(state)


def test_knapsack__step(knapsack_env: Knapsack) -> None:
    """Validates the jitted step of the environment."""
    chex.clear_trace_counter()

    step_fn = chex.assert_max_traces(knapsack_env.step, n=1)
    step_fn = jax.jit(step_fn)

    key = random.PRNGKey(0)
    reset_key, step_key = jax.random.split(key)
    state, timestep = knapsack_env.reset(reset_key)

    action = jax.random.randint(
        step_key, shape=(), minval=0, maxval=knapsack_env.num_items
    )
    new_state, next_timestep = step_fn(state, action)

    # Check that the state has changed
    assert not jnp.array_equal(new_state.packed_items, state.packed_items)
    assert not jnp.array_equal(new_state.remaining_budget, state.remaining_budget)

    # Check token was inserted as expected
    assert new_state.packed_items[action] == 1
    assert new_state.packed_items.sum() == 1

    # New step with same action should be invalid
    state = new_state

    new_state, next_timestep = step_fn(state, action)

    # Check that the state has not changed
    assert jnp.array_equal(new_state.packed_items, state.packed_items)
    assert jnp.array_equal(new_state.remaining_budget, state.remaining_budget)


def test_knapsack__does_not_smoke(knapsack_env: Knapsack) -> None:
    """Test that we can run an episode without any errors."""
    check_env_does_not_smoke(knapsack_env)


def test_knapsackenv__trajectory_action(knapsack_env: Knapsack) -> None:
    """Checks that the agent stops when the remaining budget does not allow extra items
    and that the appropriate reward is received
    """
    key = random.PRNGKey(0)
    state, timestep = knapsack_env.reset(key)

    while not timestep.last():
        # check that budget remains positive
        assert state.remaining_budget > 0

        # check that the reward is 0 while trajectory is not done
        assert timestep.reward == 0

        action = jnp.argmax(timestep.observation.action_mask)

        state, timestep = knapsack_env.step(state, action)

    # check that the reward is positive when trajectory is done
    assert timestep.reward > 0

    # check that no action can be taken: remaining items are too large
    assert knapsack_env._state_to_observation(state).action_mask.sum() == 0
    assert timestep.last()


def test_knapsackenv__invalid_action(knapsack_env: Knapsack) -> None:
    """Checks that an invalid action leads to a termination
    and the appropriate reward is received
    """
    key = random.PRNGKey(0)
    reset_key, item_key = jax.random.split(key, 2)
    state, timestep = knapsack_env.reset(reset_key)

    first_item = jax.random.randint(
        item_key, shape=(), minval=0, maxval=knapsack_env.num_items
    )
    actions = (
        jnp.array([first_item + 1, first_item + 2, first_item + 2])
        % knapsack_env.num_items
    )

    for a in actions:
        assert timestep.reward == 0
        assert timestep.step_type < StepType.LAST
        state, timestep = knapsack_env.step(state, a)

    # last action is invalid because it was already taken
    assert timestep.reward > 0
    assert timestep.last()
