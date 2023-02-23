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
import jax.numpy as jnp
import pytest

from jumanji.environments.routing.tsp.env import TSP
from jumanji.environments.routing.tsp.types import State
from jumanji.testing.env_not_smoke import check_env_does_not_smoke
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import StepType, TimeStep


class TestDenseTSP:
    def test_tsp_dense__reset(self, tsp_dense_reward: TSP) -> None:
        """Validates the jitted reset of the environment."""
        reset_fn = jax.jit(tsp_dense_reward.reset)
        key = jax.random.PRNGKey(0)
        state, timestep = reset_fn(key)
        assert isinstance(timestep, TimeStep)
        assert isinstance(state, State)
        assert state.position == -1
        assert jnp.all(state.visited_mask == 0)
        assert jnp.all(state.trajectory == -1)
        assert state.num_visited == 0

        assert_is_jax_array_tree(state)

    def test_tsp_dense__step(self, tsp_dense_reward: TSP) -> None:
        """Validates the jitted step of the environment."""
        chex.clear_trace_counter()

        step_fn = chex.assert_max_traces(tsp_dense_reward.step, n=1)
        step_fn = jax.jit(step_fn)

        key = jax.random.PRNGKey(0)
        reset_key, step_key = jax.random.split(key)
        state, timestep = tsp_dense_reward.reset(reset_key)

        action = jax.random.randint(
            step_key, shape=(), minval=0, maxval=tsp_dense_reward.num_cities
        )
        new_state, next_timestep = step_fn(state, action)

        # Check that the state has changed
        assert not jnp.array_equal(new_state.position, state.position)
        assert not jnp.array_equal(new_state.visited_mask, state.visited_mask)
        assert not jnp.array_equal(new_state.trajectory, state.trajectory)
        assert not jnp.array_equal(new_state.num_visited, state.num_visited)

        # Check that the state is made of DeviceArrays, this is false for the non-jitted
        # step function since unpacking random.split returns numpy arrays and not device arrays.
        assert_is_jax_array_tree(new_state)

        # Check token was inserted as expected
        assert new_state.visited_mask[action] == 1
        assert new_state.visited_mask.sum() == 1

        # New step with same action should be invalid
        state = new_state

        new_state, next_timestep = step_fn(state, action)

        # Check that the state has not changed
        assert jnp.array_equal(new_state.position, state.position)
        assert jnp.array_equal(new_state.visited_mask, state.visited_mask)
        assert jnp.array_equal(new_state.trajectory, state.trajectory)
        assert jnp.array_equal(new_state.num_visited, state.num_visited)

    def test_tsp_dense__does_not_smoke(
        self, tsp_dense_reward: TSP, capsys: pytest.CaptureFixture
    ) -> None:
        """Test that we can run an episode without any errors."""
        check_env_does_not_smoke(tsp_dense_reward)

    def test_tsp_dense__trajectory_action(self, tsp_dense_reward: TSP) -> None:
        """Checks that the agent stops when there are no more cities to be selected and that the
        appropriate reward is received. The testing loop ensures that no city is selected twice.
        """
        key = jax.random.PRNGKey(0)
        state, timestep = tsp_dense_reward.reset(key)

        while not timestep.last():
            # Check that there are cities that have not been selected yet.
            assert state.num_visited < tsp_dense_reward.num_cities
            assert state.visited_mask.sum() < tsp_dense_reward.num_cities
            state, timestep = tsp_dense_reward.step(
                state, jnp.argmax(timestep.observation.action_mask)
            )
            # Check that the reward is negative for all but the first city
            if state.num_visited == 1:
                assert timestep.reward == 0
            else:
                assert timestep.reward < 0

        # Check that no action can be taken (all cities have been selected)
        assert state.num_visited == tsp_dense_reward.num_cities
        assert state.visited_mask.sum() == tsp_dense_reward.num_cities
        assert timestep.last()

    def test_tsp_dense__invalid_action(self, tsp_dense_reward: TSP) -> None:
        """Checks that an invalid action leads to a termination and the appropriate reward is
        received.
        """
        key = jax.random.PRNGKey(73)
        reset_key, position_key = jax.random.split(key, 2)
        state, timestep = tsp_dense_reward.reset(reset_key)

        first_position = jax.random.randint(
            position_key, shape=(), minval=0, maxval=tsp_dense_reward.num_cities
        )
        actions = (
            jnp.array([first_position + 1, first_position + 2, first_position + 2])
            % tsp_dense_reward.num_cities
        )

        for a in actions:
            assert timestep.step_type < StepType.LAST
            state, timestep = tsp_dense_reward.step(state, a)
            assert timestep.reward < 0 or state.num_visited == 1

        # Last action is invalid because it was already taken
        assert timestep.reward == -tsp_dense_reward.num_cities * jnp.sqrt(2)
        assert timestep.last()


class TestSparseTSP:
    def test_tsp_sparse__reset(self, tsp_sparse_reward: TSP) -> None:
        """Validates the jitted reset of the environment."""
        reset_fn = jax.jit(tsp_sparse_reward.reset)
        key = jax.random.PRNGKey(0)
        state, timestep = reset_fn(key)
        assert isinstance(timestep, TimeStep)
        assert isinstance(state, State)
        assert state.position == -1
        assert jnp.all(state.visited_mask == 0)
        assert jnp.all(state.trajectory == -1)
        assert state.num_visited == 0
        assert_is_jax_array_tree(state)

    def test_tsp_sparse__step(self, tsp_sparse_reward: TSP) -> None:
        """Validates the environment step and that it is jit-ed only once."""
        chex.clear_trace_counter()

        step_fn = chex.assert_max_traces(tsp_sparse_reward.step, n=1)
        step_fn = jax.jit(step_fn)

        key = jax.random.PRNGKey(0)
        reset_key, step_key = jax.random.split(key)
        state, timestep = tsp_sparse_reward.reset(reset_key)

        action = jax.random.randint(
            step_key, shape=(), minval=0, maxval=tsp_sparse_reward.num_cities
        )
        new_state, next_timestep = step_fn(state, action)

        # Check that the state has changed
        assert not jnp.array_equal(new_state.position, state.position)
        assert not jnp.array_equal(new_state.visited_mask, state.visited_mask)
        assert not jnp.array_equal(new_state.trajectory, state.trajectory)
        assert not jnp.array_equal(new_state.num_visited, state.num_visited)

        # Check that the state is made of DeviceArrays, this is false for the non-jitted
        # step function since unpacking random.split returns numpy arrays and not device arrays.
        assert_is_jax_array_tree(new_state)

        # Check token was inserted as expected
        assert new_state.visited_mask[action] == 1
        assert new_state.visited_mask.sum() == 1

        # New step with same action should be invalid
        state = new_state

        new_state, next_timestep = step_fn(state, action)

        # Check that the state has not changed
        assert jnp.array_equal(new_state.position, state.position)
        assert jnp.array_equal(new_state.visited_mask, state.visited_mask)
        assert jnp.array_equal(new_state.trajectory, state.trajectory)
        assert jnp.array_equal(new_state.num_visited, state.num_visited)

    def test_tsp_sparse__does_not_smoke(
        self, tsp_sparse_reward: TSP, capsys: pytest.CaptureFixture
    ) -> None:
        """Test that we can run an episode without any errors."""
        check_env_does_not_smoke(tsp_sparse_reward)

    def test_tsp_sparse__trajectory_action(self, tsp_sparse_reward: TSP) -> None:
        """Checks that the agent stops when there are no more cities to be selected and that the
        appropriate reward is received. The testing loop ensures that no city is selected twice.
        """
        key = jax.random.PRNGKey(0)
        state, timestep = tsp_sparse_reward.reset(key)

        while not timestep.last():
            # Check that there are cities that have not been selected yet.
            assert state.num_visited < tsp_sparse_reward.num_cities
            assert state.visited_mask.sum() < tsp_sparse_reward.num_cities
            # Check that the reward is 0 while trajectory is not done.
            assert timestep.reward == 0
            state, timestep = tsp_sparse_reward.step(
                state, jnp.argmax(timestep.observation.action_mask)
            )

        # Check that the reward is negative when trajectory is done.
        assert timestep.reward < 0

        # Check that no action can be taken (all cities have been selected)
        assert state.num_visited == tsp_sparse_reward.num_cities
        assert state.visited_mask.sum() == tsp_sparse_reward.num_cities
        assert timestep.last()

    def test_tsp_sparse__invalid_action(self, tsp_sparse_reward: TSP) -> None:
        """Checks that an invalid action leads to a termination and the appropriate reward is
        received.
        """
        key = jax.random.PRNGKey(0)
        reset_key, position_key = jax.random.split(key, 2)
        state, timestep = tsp_sparse_reward.reset(reset_key)

        first_position = jax.random.randint(
            position_key, shape=(), minval=0, maxval=tsp_sparse_reward.num_cities
        )
        actions = (
            jnp.array([first_position + 1, first_position + 2, first_position + 2])
            % tsp_sparse_reward.num_cities
        )

        for a in actions:
            assert timestep.reward == 0
            assert timestep.step_type < StepType.LAST
            state, timestep = tsp_sparse_reward.step(state, a)

        # Last action is invalid because it was already taken
        assert timestep.reward == -tsp_sparse_reward.num_cities * jnp.sqrt(2)
        assert timestep.last()


def test_tsp__equivalence_dense_sparse_reward(
    tsp_dense_reward: TSP, tsp_sparse_reward: TSP
) -> None:
    key = jax.random.PRNGKey(0)

    # Dense reward
    state, timestep = tsp_dense_reward.reset(key)
    return_dense = timestep.reward
    while not timestep.last():
        state, timestep = tsp_dense_reward.step(state, jnp.argmin(state.visited_mask))
        return_dense += timestep.reward

    # Sparse reward
    state, timestep = tsp_sparse_reward.reset(key)
    return_sparse = timestep.reward
    while not timestep.last():
        state, timestep = tsp_sparse_reward.step(state, jnp.argmin(state.visited_mask))
        return_sparse += timestep.reward

    # Check that both returns are the same and not the invalid action penalty
    assert return_sparse == return_dense != -tsp_dense_reward.num_cities * jnp.sqrt(2)
