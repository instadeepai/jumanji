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
from jax import numpy as jnp

from jumanji.environments.packing.knapsack import Knapsack, State
from jumanji.testing.env_not_smoke import (
    check_env_does_not_smoke,
    check_env_specs_does_not_smoke,
)
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import StepType, TimeStep


class TestSparseKnapsack:
    def test_knapsack_sparse__reset(self, knapsack_sparse_reward: Knapsack) -> None:
        """Validates the jitted reset of the environment."""
        chex.clear_trace_counter()
        reset_fn = jax.jit(chex.assert_max_traces(knapsack_sparse_reward.reset, n=1))

        key = jax.random.PRNGKey(0)
        state, timestep = reset_fn(key)
        # Call again to check it does not compile twice.
        state, timestep = reset_fn(key)

        assert isinstance(timestep, TimeStep)
        assert isinstance(state, State)
        assert state.remaining_budget == knapsack_sparse_reward.total_budget > 0
        assert not jnp.any(state.packed_items)

        # Check that the state is made of DeviceArrays, this is false for the non-jitted
        # reset function since unpacking random.split returns numpy arrays and not device arrays.
        assert_is_jax_array_tree(state)

    def test_knapsack_sparse__step(self, knapsack_sparse_reward: Knapsack) -> None:
        """Validates the jitted step of the environment."""
        chex.clear_trace_counter()
        step_fn = jax.jit(chex.assert_max_traces(knapsack_sparse_reward.step, n=1))

        key = jax.random.PRNGKey(0)
        state, timestep = knapsack_sparse_reward.reset(key)

        action = jax.random.randint(
            key, shape=(), minval=0, maxval=knapsack_sparse_reward.num_items
        )
        new_state, next_timestep = step_fn(state, action)

        # Check that the state has changed.
        assert not jnp.array_equal(new_state.packed_items, state.packed_items)
        assert not jnp.array_equal(new_state.remaining_budget, state.remaining_budget)

        # Check token was inserted as expected.
        assert new_state.packed_items[action]
        assert new_state.packed_items.sum() == 1

        # Check that the state does not change when taking the same action again (invalid).
        state = new_state
        new_state, next_timestep = step_fn(state, action)
        assert jnp.array_equal(new_state.packed_items, state.packed_items)
        assert jnp.array_equal(new_state.remaining_budget, state.remaining_budget)

    def test_knapsack_sparse__does_not_smoke(
        self, knapsack_sparse_reward: Knapsack
    ) -> None:
        """Test that we can run an episode without any errors."""
        check_env_does_not_smoke(knapsack_sparse_reward)

    def test_knapsack_sparse__specs_does_not_smoke(
        self, knapsack_sparse_reward: Knapsack
    ) -> None:
        """Test that we can access specs without any errors."""
        check_env_specs_does_not_smoke(knapsack_sparse_reward)

    def test_knapsack_sparse__trajectory_action(
        self, knapsack_sparse_reward: Knapsack
    ) -> None:
        """Checks that the agent stops when the remaining budget does not allow extra items
        and that the appropriate reward is received.
        """
        step_fn = jax.jit(knapsack_sparse_reward.step)
        key = jax.random.PRNGKey(0)
        state, timestep = knapsack_sparse_reward.reset(key)

        while not timestep.last():
            # Check that the budget remains positive.
            assert state.remaining_budget > 0

            # Check that the reward is 0 while trajectory is not done.
            assert timestep.reward == 0

            action = jnp.argmax(timestep.observation.action_mask)
            state, timestep = step_fn(state, action)

        # Check that the reward is positive when the trajectory is done.
        assert timestep.reward > 0

        # Check that no action can be taken: the remaining items are too large.
        assert not jnp.any(timestep.observation.action_mask)
        assert timestep.last()

    def test_knapsack_sparse__invalid_action(
        self, knapsack_sparse_reward: Knapsack
    ) -> None:
        """Checks that an invalid action leads to a termination and that the appropriate reward
        is returned.
        """
        step_fn = jax.jit(knapsack_sparse_reward.step)
        key = jax.random.PRNGKey(0)
        state, timestep = knapsack_sparse_reward.reset(key)

        first_item = jax.random.randint(
            key, shape=(), minval=0, maxval=knapsack_sparse_reward.num_items
        )
        actions = (
            jnp.array([first_item + 1, first_item + 2, first_item + 2])
            % knapsack_sparse_reward.num_items
        )

        for a in actions:
            assert timestep.reward == 0
            assert timestep.step_type < StepType.LAST
            state, timestep = step_fn(state, a)

        # last action is invalid because it was already taken
        assert timestep.reward == 0
        assert timestep.last()


class TestDenseKnapsack:
    def test_knapsack_dense__reset(self, knapsack_dense_reward: Knapsack) -> None:
        """Validates the jitted reset of the environment."""
        chex.clear_trace_counter()
        reset_fn = jax.jit(chex.assert_max_traces(knapsack_dense_reward.reset, n=1))

        key = jax.random.PRNGKey(0)
        state, timestep = reset_fn(key)
        # Call again to check it does not compile twice.
        state, timestep = reset_fn(key)

        assert isinstance(timestep, TimeStep)
        assert isinstance(state, State)
        assert state.remaining_budget == knapsack_dense_reward.total_budget > 0
        assert not jnp.any(state.packed_items)

        # Check that the state is made of DeviceArrays, this is false for the non-jitted
        # reset function since unpacking random.split returns numpy arrays and not device arrays.
        assert_is_jax_array_tree(state)

    def test_knapsack_dense__step(self, knapsack_dense_reward: Knapsack) -> None:
        """Validates the jitted step of the environment."""
        chex.clear_trace_counter()
        step_fn = jax.jit(chex.assert_max_traces(knapsack_dense_reward.step, n=1))

        key = jax.random.PRNGKey(0)
        state, timestep = knapsack_dense_reward.reset(key)

        action = jax.random.randint(
            key, shape=(), minval=0, maxval=knapsack_dense_reward.num_items
        )
        new_state, next_timestep = step_fn(state, action)

        # Check that the state has changed.
        assert not jnp.array_equal(new_state.packed_items, state.packed_items)
        assert not jnp.array_equal(new_state.remaining_budget, state.remaining_budget)

        # Check token was inserted as expected.
        assert new_state.packed_items[action]
        assert new_state.packed_items.sum() == 1

        # Check that the state does not change when taking the same action again (invalid).
        state = new_state
        new_state, next_timestep = step_fn(state, action)
        assert jnp.array_equal(new_state.packed_items, state.packed_items)
        assert jnp.array_equal(new_state.remaining_budget, state.remaining_budget)

    def test_knapsack_dense__does_not_smoke(
        self, knapsack_dense_reward: Knapsack
    ) -> None:
        """Test that we can run an episode without any errors."""
        check_env_does_not_smoke(knapsack_dense_reward)

    def test_knapsack_dense__trajectory_action(
        self, knapsack_dense_reward: Knapsack
    ) -> None:
        """Checks that the agent stops when the remaining budget does not allow extra items
        and that the appropriate reward is received.
        """
        step_fn = jax.jit(knapsack_dense_reward.step)
        key = jax.random.PRNGKey(0)
        state, timestep = knapsack_dense_reward.reset(key)

        while not timestep.last():
            # Check that the budget remains positive.
            assert state.remaining_budget > 0

            # Check that the reward is positive at each but the first step.
            assert timestep.reward > 0 or timestep.first()

            action = jnp.argmax(timestep.observation.action_mask)
            state, timestep = step_fn(state, action)

        # Check that the reward is also positive when the trajectory is done.
        assert timestep.reward > 0

        # Check that no action can be taken: the remaining items are too large.
        assert not jnp.any(timestep.observation.action_mask)
        assert timestep.last()

    def test_knapsack_dense__invalid_action(
        self, knapsack_dense_reward: Knapsack
    ) -> None:
        """Checks that an invalid action leads to a termination and that the appropriate reward
        is returned.
        """
        step_fn = jax.jit(knapsack_dense_reward.step)
        key = jax.random.PRNGKey(0)
        state, timestep = knapsack_dense_reward.reset(key)

        first_item = jax.random.randint(
            key, shape=(), minval=0, maxval=knapsack_dense_reward.num_items
        )
        actions = (
            jnp.array([first_item + 1, first_item + 2, first_item + 2])
            % knapsack_dense_reward.num_items
        )

        for a in actions:
            assert timestep.reward > 0 or timestep.first()
            assert timestep.step_type < StepType.LAST
            state, timestep = step_fn(state, a)

        # last action is invalid because it was already taken
        assert timestep.reward == 0
        assert timestep.last()


def test_knapsack__equivalence_dense_sparse_reward(
    knapsack_dense_reward: Knapsack, knapsack_sparse_reward: Knapsack
) -> None:
    dense_step_fn = jax.jit(knapsack_dense_reward.step)
    sparse_step_fn = jax.jit(knapsack_sparse_reward.step)
    key = jax.random.PRNGKey(0)

    # Dense reward
    state, timestep = knapsack_dense_reward.reset(key)
    return_dense = timestep.reward
    while not timestep.last():
        action = jnp.argmax(timestep.observation.action_mask)
        state, timestep = dense_step_fn(state, action)
        return_dense += timestep.reward

    # Sparse reward
    state, timestep = knapsack_sparse_reward.reset(key)
    return_sparse = timestep.reward
    while not timestep.last():
        action = jnp.argmax(timestep.observation.action_mask)
        state, timestep = sparse_step_fn(state, action)
        return_sparse += timestep.reward

    # Check that both returns are the same and not the invalid action penalty
    assert return_sparse == return_dense > 0
