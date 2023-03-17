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

from jumanji.environments.routing.cvrp.constants import DEPOT_IDX
from jumanji.environments.routing.cvrp.env import CVRP
from jumanji.environments.routing.cvrp.types import State
from jumanji.testing.env_not_smoke import check_env_does_not_smoke
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import TimeStep


class TestSparseCVRP:
    def test_cvrp_sparse__reset(self, cvrp_sparse_reward: CVRP) -> None:
        """Validates the jitted reset of the environment."""
        chex.clear_trace_counter()
        reset_fn = jax.jit(chex.assert_max_traces(cvrp_sparse_reward.reset, n=1))

        key = jax.random.PRNGKey(0)
        state, timestep = reset_fn(key)
        # Call again to check it does not compile twice.
        state, timestep = reset_fn(key)

        assert isinstance(timestep, TimeStep)
        assert isinstance(state, State)

        # Initial position is at depot, so current capacity is max capacity.
        assert state.capacity == cvrp_sparse_reward.max_capacity
        # The depot is initially visited.
        assert state.visited_mask[DEPOT_IDX]
        assert state.visited_mask.sum() == 1
        # First visited position is the depot.
        assert state.trajectory[0] == DEPOT_IDX
        assert state.num_total_visits == 1
        # Cost of the depot must be 0.
        assert state.demands[DEPOT_IDX] == 0

        assert_is_jax_array_tree(state)

    def test_cvrp_sparse__step(self, cvrp_sparse_reward: CVRP) -> None:
        """Validates the jitted step of the environment."""
        chex.clear_trace_counter()

        step_fn = chex.assert_max_traces(cvrp_sparse_reward.step, n=1)
        step_fn = jax.jit(step_fn)

        key = jax.random.PRNGKey(0)
        state, timestep = cvrp_sparse_reward.reset(key)

        # Starting position is depot, first action to visit first node.
        new_action = 1
        new_state, next_timestep = step_fn(state, new_action)

        # Check that the state has changed.
        assert not jnp.array_equal(new_state.position, state.position)
        assert not jnp.array_equal(new_state.capacity, state.capacity)
        assert not jnp.array_equal(new_state.visited_mask, state.visited_mask)
        assert not jnp.array_equal(new_state.trajectory, state.trajectory)
        assert not jnp.array_equal(new_state.num_total_visits, state.num_total_visits)

        # Check that the state is made of DeviceArrays, this is false for the non-jitted
        # step function since unpacking random.split returns numpy arrays and not device arrays.
        assert_is_jax_array_tree(new_state)

        # Check the state was changed as expected.
        assert new_state.visited_mask[new_action]
        assert new_state.visited_mask.sum() == 1

        # New step with same action should be invalid.
        state = new_state

        new_state, next_timestep = step_fn(state, new_action)

        # Check that the state has not changed.
        assert jnp.array_equal(new_state.position, state.position)
        assert jnp.array_equal(new_state.visited_mask, state.visited_mask)
        assert jnp.array_equal(new_state.trajectory, state.trajectory)
        assert jnp.array_equal(new_state.num_total_visits, state.num_total_visits)

    def test_cvrp_sparse__does_not_smoke(self, cvrp_sparse_reward: CVRP) -> None:
        """Test that we can run an episode without any errors."""
        check_env_does_not_smoke(cvrp_sparse_reward)

    def test_cvrp_sparse__trajectory_action(self, cvrp_sparse_reward: CVRP) -> None:
        """Tests a trajectory by visiting nodes in increasing and cyclic order, visiting the depot
        when the next node in the list surpasses the current capacity of the agent.
        """
        step_fn = jax.jit(cvrp_sparse_reward.step)
        key = jax.random.PRNGKey(0)
        state, timestep = cvrp_sparse_reward.reset(key)

        # The position from which we will continue visiting after we recharge capacity at the depot.
        pending_position = None
        while not timestep.last():
            # Check that there are nodes that have not been selected yet.
            assert not state.visited_mask.all()

            # Check that the reward is 0 while the trajectory is not done.
            assert timestep.reward == 0

            if pending_position is not None:
                next_position = pending_position
                pending_position = None
            else:
                # Select the next position. If we don't have enough capacity, set it
                # as pending and go to the depot (0) to recharge.
                next_position = (state.position % cvrp_sparse_reward.num_nodes) + 1
                if state.capacity < state.demands[next_position]:
                    pending_position = next_position
                    next_position = DEPOT_IDX

            # If all cities have been visited, go to depot to finish the episode.
            visited_nodes_other_than_depot = jnp.concatenate(
                [state.visited_mask[:DEPOT_IDX], state.visited_mask[DEPOT_IDX + 1 :]],
                axis=0,
            )
            if visited_nodes_other_than_depot.all():
                next_position = DEPOT_IDX

            state, timestep = step_fn(state, next_position)

        # Check that the reward is negative when the trajectory is done.
        assert timestep.reward < 0
        assert timestep.last()

        # Check that no action can be taken (all nodes have been selected).
        assert state.visited_mask.all()

    def test_cvrp_sparse__invalid_revisit_node(self, cvrp_sparse_reward: CVRP) -> None:
        """Checks that an invalid action leads to a termination and the appropriate reward is
        received.
        """
        step_fn = jax.jit(cvrp_sparse_reward.step)
        key = jax.random.PRNGKey(0)
        state, timestep = cvrp_sparse_reward.reset(key)

        first_position = state.position
        actions = (
            jnp.array([first_position + 1, first_position + 2, first_position + 2])
            % cvrp_sparse_reward.num_nodes
        )

        for a in actions:
            assert timestep.reward == 0
            assert not timestep.last()
            state, timestep = step_fn(state, a)

        # Last action is invalid because it was already taken.
        assert timestep.reward < 0
        assert timestep.last()

    def test_cvrp_sparse__revisit_depot(self, cvrp_sparse_reward: CVRP) -> None:
        """Checks that the depot can be revisited and that the capacity is set back to the maximum
        when visited.
        """
        step_fn = jax.jit(cvrp_sparse_reward.step)
        key = jax.random.PRNGKey(0)
        state, timestep = cvrp_sparse_reward.reset(key)
        state, timestep = step_fn(state, action=1)

        # Make sure we are not at the depot.
        assert state.position != DEPOT_IDX
        # We have moved from the depot, so capacity must be lower than 1.0.
        assert state.capacity < cvrp_sparse_reward.max_capacity

        state, timestep = step_fn(state, DEPOT_IDX)

        assert state.position == DEPOT_IDX
        assert state.capacity == cvrp_sparse_reward.max_capacity
        assert timestep.reward == 0
        assert not timestep.last()

    def test_cvrp_sparse__revisit_depot_invalid(self, cvrp_sparse_reward: CVRP) -> None:
        """Checks that the depot cannot be revisited if we are already at the depot."""
        key = jax.random.PRNGKey(0)
        state, timestep = cvrp_sparse_reward.reset(key)
        state, timestep = cvrp_sparse_reward.step(
            state, jnp.array(DEPOT_IDX, jnp.int32)
        )

        assert timestep.last()


class TestDenseCVRP:
    def test_cvrp_dense__reset(self, cvrp_dense_reward: CVRP) -> None:
        """Validates the jitted reset of the environment."""
        chex.clear_trace_counter()
        reset_fn = jax.jit(chex.assert_max_traces(cvrp_dense_reward.reset, n=1))

        key = jax.random.PRNGKey(0)
        state, timestep = reset_fn(key)
        # Call again to check it does not compile twice.
        state, timestep = reset_fn(key)

        assert isinstance(timestep, TimeStep)
        assert isinstance(state, State)

        # Initial position is at depot, so current capacity is max capacity.
        assert state.capacity == cvrp_dense_reward.max_capacity
        # # The depot is initially visited.
        assert state.visited_mask[DEPOT_IDX]
        assert state.visited_mask.sum() == 1
        # First visited position is the depot.
        assert state.trajectory[0] == DEPOT_IDX
        assert state.num_total_visits == 1
        # Cost of the depot must be 0.
        assert state.demands[DEPOT_IDX] == 0

        assert_is_jax_array_tree(state)

    def test_cvrp_dense__step(self, cvrp_dense_reward: CVRP) -> None:
        """Validates the jitted step of the environment."""
        chex.clear_trace_counter()

        step_fn = chex.assert_max_traces(cvrp_dense_reward.step, n=1)
        step_fn = jax.jit(step_fn)

        key = jax.random.PRNGKey(0)
        state, timestep = cvrp_dense_reward.reset(key)

        # Starting position is depot, new action to visit first node.
        new_action = 1
        new_state, next_timestep = step_fn(state, new_action)

        # Check that the state has changed.
        assert not jnp.array_equal(new_state.position, state.position)
        assert not jnp.array_equal(new_state.capacity, state.capacity)
        assert not jnp.array_equal(new_state.visited_mask, state.visited_mask)
        assert not jnp.array_equal(new_state.trajectory, state.trajectory)
        assert not jnp.array_equal(new_state.num_total_visits, state.num_total_visits)

        # Check that the state is made of DeviceArrays, this is false for the non-jitted
        # step function since unpacking random.split returns numpy arrays and not device arrays.
        assert_is_jax_array_tree(new_state)

        # Check the state was changed as expected.
        assert new_state.visited_mask[new_action]
        assert new_state.visited_mask.sum() == 1

        # New step with same action should be invalid.
        state = new_state

        new_state, next_timestep = step_fn(state, new_action)

        # Check that the state has not changed.
        assert jnp.array_equal(new_state.position, state.position)
        assert jnp.array_equal(new_state.visited_mask, state.visited_mask)
        assert jnp.array_equal(new_state.trajectory, state.trajectory)
        assert jnp.array_equal(new_state.num_total_visits, state.num_total_visits)

    def test_cvrp_dense__does_not_smoke(self, cvrp_dense_reward: CVRP) -> None:
        """Test that we can run an episode without any errors."""
        check_env_does_not_smoke(cvrp_dense_reward)

    def test_cvrp_dense__trajectory_action(self, cvrp_dense_reward: CVRP) -> None:
        """Tests a trajectory by visiting nodes in increasing and cyclic order, visiting the depot
        when the next node in the list surpasses the current capacity of the agent.
        """
        step_fn = jax.jit(cvrp_dense_reward.step)
        key = jax.random.PRNGKey(0)
        state, timestep = cvrp_dense_reward.reset(key)

        # The position from which we will continue visiting after we recharge capacity at the depot.
        pending_position = None
        while not timestep.last():
            # Check that there are nodes that have not been selected yet.
            assert not state.visited_mask.all()

            # Check that the reward is always negative.
            assert timestep.reward < 0 or timestep.first()

            if pending_position is not None:
                next_position = pending_position
                pending_position = None
            else:
                # Select the next position. If we don't have enough capacity, set it
                # as pending and go to the depot (0) to recharge.
                next_position = (state.position % cvrp_dense_reward.num_nodes) + 1
                if state.capacity < state.demands[next_position]:
                    pending_position = next_position
                    next_position = DEPOT_IDX

            # If all cities have been visited, go to depot to finish the episode.
            visited_nodes_other_than_depot = jnp.concatenate(
                [state.visited_mask[:DEPOT_IDX], state.visited_mask[DEPOT_IDX + 1 :]],
                axis=0,
            )
            if visited_nodes_other_than_depot.all():
                next_position = DEPOT_IDX

            state, timestep = step_fn(state, next_position)

        # Check that the reward is negative when the trajectory is done as well.
        assert timestep.reward < 0
        assert timestep.last()

        # Check that no action can be taken (all nodes have been selected).
        assert state.visited_mask.all()

    def test_cvrp_dense__invalid_revisit_node(self, cvrp_dense_reward: CVRP) -> None:
        """Checks that an invalid action leads to a termination and the appropriate reward is
        received.
        """
        step_fn = jax.jit(cvrp_dense_reward.step)
        key = jax.random.PRNGKey(0)
        state, timestep = cvrp_dense_reward.reset(key)

        first_position = state.position
        actions = (
            jnp.array([first_position + 1, first_position + 2, first_position + 2])
            % cvrp_dense_reward.num_nodes
        )

        for a in actions:
            assert not timestep.last()
            state, timestep = step_fn(state, a)
            assert timestep.reward < 0

        # Last action is invalid because it was already taken.
        assert timestep.last()

    def test_cvrp_dense__revisit_depot(self, cvrp_dense_reward: CVRP) -> None:
        """Checks that the depot can be revisited and that the capacity is set back to the maximum
        when visited.
        """
        step_fn = jax.jit(cvrp_dense_reward.step)
        key = jax.random.PRNGKey(0)
        state, timestep = cvrp_dense_reward.reset(key)
        state, timestep = step_fn(state, action=1)

        # Make sure we are not at the depot.
        assert state.position != DEPOT_IDX
        # We have moved from the depot, so capacity must be lower than 1.0.
        assert state.capacity < cvrp_dense_reward.max_capacity

        state, timestep = step_fn(state, DEPOT_IDX)

        assert state.position == DEPOT_IDX
        assert state.capacity == cvrp_dense_reward.max_capacity
        assert timestep.reward < 0
        assert not timestep.last()

    def test_cvrp_dense__revisit_depot_invalid(self, cvrp_dense_reward: CVRP) -> None:
        """Checks that the depot cannot be revisited if we are already at the depot."""
        key = jax.random.PRNGKey(0)
        state, timestep = cvrp_dense_reward.reset(key)
        state, timestep = cvrp_dense_reward.step(state, DEPOT_IDX)

        assert timestep.last()


def test_cvrp__equivalence_dense_sparse_reward(
    cvrp_dense_reward: CVRP, cvrp_sparse_reward: CVRP
) -> None:
    dense_step_fn = jax.jit(cvrp_dense_reward.step)
    sparse_step_fn = jax.jit(cvrp_sparse_reward.step)
    key = jax.random.PRNGKey(0)

    # Dense reward
    state, timestep = cvrp_dense_reward.reset(key)
    return_dense = timestep.reward
    while not timestep.last():
        state, timestep = dense_step_fn(state, jnp.argmin(state.visited_mask))
        return_dense += timestep.reward

    # Sparse reward
    state, timestep = cvrp_sparse_reward.reset(key)
    return_sparse = timestep.reward
    while not timestep.last():
        state, timestep = sparse_step_fn(state, jnp.argmin(state.visited_mask))
        return_sparse += timestep.reward

    # Check that both returns are the same and not the invalid action penalty
    assert (
        return_sparse == return_dense > -2 * cvrp_dense_reward.num_nodes * jnp.sqrt(2)
    )
