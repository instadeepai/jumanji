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

from jumanji.environments.combinatorial.cvrp.env import CVRP
from jumanji.environments.combinatorial.cvrp.types import State
from jumanji.environments.combinatorial.cvrp.utils import (
    DEPOT_IDX,
    compute_tour_length,
    get_augmentations,
)

from jumanji.testing.env_not_smoke import check_env_does_not_smoke
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import StepType, TimeStep


@pytest.fixture
def cvrp_env() -> CVRP:
    """Instantiates a default CVRP environment."""
    return CVRP()


def test_cvrp__reset(cvrp_env: CVRP) -> None:
    """Validates the jitted reset of the environment."""
    chex.clear_trace_counter()
    reset_fn = jax.jit(chex.assert_max_traces(cvrp_env.reset, n=1))

    key = jax.random.PRNGKey(0)
    state, timestep = reset_fn(key)
    # Call again to check it does not compile twice.
    state, timestep = reset_fn(key)

    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)

    # Initial position is at depot, so current capacity is max capacity
    assert (state.capacity == cvrp_env.max_capacity)
    # # The depot is initially visited
    assert state.visited_mask[DEPOT_IDX]
    assert state.visited_mask.sum() == 1
    # First visited position is the depot
    assert state.order[0] == DEPOT_IDX
    assert state.num_total_visits == 1
    # Cost of the depot must be 0
    assert state.demands[DEPOT_IDX] == 0

    assert_is_jax_array_tree(state)


def test_cvrp__step(cvrp_env: CVRP) -> None:
    """Validates the jitted step of the environment."""
    chex.clear_trace_counter()

    step_fn = chex.assert_max_traces(cvrp_env.step, n=1)
    step_fn = jax.jit(step_fn)

    key = jax.random.PRNGKey(0)
    state, timestep = cvrp_env.reset(key)

    # Starting position is depot, new action to visit first node
    new_action = 1
    new_state, next_timestep = step_fn(state, new_action)

    # Check that the state has changed
    assert not jnp.array_equal(new_state.position, state.position)
    assert not jnp.array_equal(new_state.capacity, state.capacity)
    assert not jnp.array_equal(new_state.visited_mask, state.visited_mask)
    assert not jnp.array_equal(new_state.order, state.order)
    assert not jnp.array_equal(new_state.num_total_visits, state.num_total_visits)

    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # step function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(new_state)

    # Check the state was changed as expected
    assert new_state.visited_mask[new_action]
    assert new_state.visited_mask.sum() == 1

    # New step with same action should be invalid
    state = new_state

    new_state, next_timestep = step_fn(state, new_action)

    # Check that the state has not changed
    assert jnp.array_equal(new_state.position, state.position)
    assert jnp.array_equal(new_state.visited_mask, state.visited_mask)
    assert jnp.array_equal(new_state.order, state.order)
    assert jnp.array_equal(new_state.num_total_visits, state.num_total_visits)


def test_cvrp__does_not_smoke(cvrp_env: CVRP) -> None:
    """Test that we can run an episode without any errors."""
    check_env_does_not_smoke(cvrp_env)


def test_cvrp__trajectory_action(cvrp_env: CVRP) -> None:
    """
    Tests a trajectory by visiting nodes in increasing and cyclic order, visiting the depot when the next node in the
    list surpasses the current capacity of the agent.
    """
    key = jax.random.PRNGKey(0)
    state, timestep = cvrp_env.reset(key)

    # The position from which we will continue visiting after we recharge capacity at the depot.
    pending_position = None
    # actions = jnp.arange(1, cvrp_env.problem_size + 1)
    while not timestep.last():
        # Check that there are nodes that have not been selected yet.
        assert state.visited_mask.sum() < (cvrp_env.problem_size + 1)

        # Check that the reward is 0 while the trajectory is not done.
        assert timestep.reward == 0

        if pending_position is not None:
            next_position = pending_position
            pending_position = None
        else:
            # Select the next position. If we don't have enough capacity, set it
            # as pending and go to the depot (0) to recharge.
            next_position = (state.position % cvrp_env.problem_size) + 1
            if state.capacity < state.demands[next_position]:
                pending_position = next_position
                next_position = DEPOT_IDX

        # If all cities have been visited, go to depot to finish the episode.
        if state.visited_mask[1:].sum() == cvrp_env.problem_size:
            next_position = DEPOT_IDX

        state, timestep = cvrp_env.step(state, next_position)

    # Check that the reward is negative when the trajectory is done.
    assert timestep.reward < 0

    # Check that no action can be taken (all nodes have been selected).
    assert state.visited_mask.sum() == cvrp_env.problem_size + 1

    assert timestep.last()


def test_cvrp__invalid_revisit_node(cvrp_env: CVRP) -> None:
    """Checks that an invalid action leads to a termination and the appropriate reward is received."""
    key = jax.random.PRNGKey(0)
    state, timestep = cvrp_env.reset(key)

    first_position = state.position
    actions = (
        jnp.array([first_position + 1, first_position + 2, first_position + 2])
        % cvrp_env.problem_size
    )

    for a in actions:
        assert timestep.reward == 0
        assert not timestep.last()
        state, timestep = cvrp_env.step(state, a)

    # Last action is invalid because it was already taken
    assert timestep.reward < 0
    assert timestep.last()


def test_cvrp__revisit_depot(cvrp_env: CVRP) -> None:
    """Checks that the depot can be revisited and that the capacity is set back to the maximum."""
    key = jax.random.PRNGKey(0)
    state, timestep = cvrp_env.reset(key)
    state, timestep = cvrp_env.step(state, action=1)

    assert state.position != DEPOT_IDX  # Not at the depot
    assert (
        state.capacity < cvrp_env.max_capacity
    )  # We have moved from the depot, so capacity must be lower than 1.0

    state, timestep = cvrp_env.step(state, jnp.int32(DEPOT_IDX))

    assert state.position == DEPOT_IDX
    assert state.capacity == cvrp_env.max_capacity
    assert timestep.reward == 0
    assert not timestep.last()


def test_cvrp__revisit_depot_invalid(cvrp_env: CVRP) -> None:
    """Checks that the depot cannot be revisited if we are already at the depot."""
    key = jax.random.PRNGKey(0)
    state, timestep = cvrp_env.reset(key)
    state, timestep = cvrp_env.step(state, jnp.int32(DEPOT_IDX))

    assert timestep.last()


def test_cvrp__tour_length() -> None:
    """Checks that the tour lengths are properly computed."""
    problem = jnp.array(
        [
            [0.65948975, 0.8527372, 0.0],
            [0.18317401, 0.06975579, 0.26666668],
            [0.4064678, 0.19167936, 0.2],
            [0.92129254, 0.27006388, 0.13333334],
            [0.7105516, 0.9370967, 0.3],
            [0.5277389, 0.18168604, 0.03333334],
            [0.47508526, 0.19661963, 0.23333333],
            [0.46782017, 0.6201354, 0.06666667],
            [0.4211073, 0.5530877, 0.23333333],
            [0.94237375, 0.64736927, 0.13333334],
            [0.97507954, 0.43589878, 0.2],
        ]
    )
    order = jnp.array([0, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 0, 6, 0, 0, 0, 0, 0])
    tour_length = compute_tour_length(problem[:, :2], order)
    assert jnp.isclose(tour_length, 6.8649917)

    # Check augmentations have same tour length
    coords_aug, demands_aug = get_augmentations(problem[:, :2], problem[:, -1])
    lengths = jax.vmap(compute_tour_length, in_axes=(0, None))(coords_aug, order)
    assert jnp.allclose(lengths, jnp.ones(coords_aug.shape[0], dtype=jnp.float32) * tour_length)

    order = jnp.array([0, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 0, 6])
    assert compute_tour_length(problem[:, :2], order) == 6.8649917

    order = jnp.array([0, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 0, 6, 0])
    assert compute_tour_length(problem[:, :2], order) == 6.8649917


def test_cvrp__augmentations() -> None:
    """Checks that the augmentations of a given instance problem is computed properly."""
    problem = jnp.array(
        [[0.65, 0.85, 0.00], [0.18, 0.06, 0.27], [0.41, 0.19, 0.20], [0.92, 0.27, 0.13]]
    )

    expected_augmentations = jnp.array(
        [
            problem,
            jnp.array(
                [
                    [0.35, 0.85, 0.00],
                    [0.82, 0.06, 0.27],
                    [0.59, 0.19, 0.20],
                    [0.08, 0.27, 0.13],
                ]
            ),
            jnp.array(
                [
                    [0.65, 0.15, 0.00],
                    [0.18, 0.94, 0.27],
                    [0.41, 0.81, 0.20],
                    [0.92, 0.73, 0.13],
                ]
            ),
            jnp.array(
                [
                    [0.35, 0.15, 0.00],
                    [0.82, 0.94, 0.27],
                    [0.59, 0.81, 0.20],
                    [0.08, 0.73, 0.13],
                ]
            ),
            jnp.array(
                [
                    [0.85, 0.65, 0.00],
                    [0.06, 0.18, 0.27],
                    [0.19, 0.41, 0.20],
                    [0.27, 0.92, 0.13],
                ]
            ),
            jnp.array(
                [
                    [0.85, 0.35, 0.00],
                    [0.06, 0.82, 0.27],
                    [0.19, 0.59, 0.20],
                    [0.27, 0.08, 0.13],
                ]
            ),
            jnp.array(
                [
                    [0.15, 0.65, 0.00],
                    [0.94, 0.18, 0.27],
                    [0.81, 0.41, 0.20],
                    [0.73, 0.92, 0.13],
                ]
            ),
            jnp.array(
                [
                    [0.15, 0.35, 0.00],
                    [0.94, 0.82, 0.27],
                    [0.81, 0.59, 0.20],
                    [0.73, 0.08, 0.13],
                ]
            ),
        ]
    )

    coords_aug, demands_aug = get_augmentations(problem[:, :2], problem[:, -1])
    augmentations = jnp.concatenate((coords_aug, demands_aug), axis=2)
    assert jnp.allclose(expected_augmentations, augmentations)
