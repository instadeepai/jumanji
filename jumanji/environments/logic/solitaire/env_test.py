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

from jumanji.environments.logic.solitaire.env import Solitaire
from jumanji.environments.logic.solitaire.types import Board, State
from jumanji.testing.env_not_smoke import check_env_does_not_smoke
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import TimeStep


@pytest.fixture
def solitaire() -> Solitaire:
    """Instantiates a default Solitaire environment."""
    return Solitaire(7)


@pytest.fixture
def board(board7x7: Board) -> Board:
    """Random board."""
    return board7x7


def test_solitaire__reset_jit(solitaire: Solitaire) -> None:
    """Confirm that the reset method is only compiled once when jitted."""
    chex.clear_trace_counter()
    reset_fn = jax.jit(chex.assert_max_traces(solitaire.reset, n=1))
    key = jax.random.PRNGKey(0)
    state, timestep = reset_fn(key)

    # Verify the data type of the output.
    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)

    # Check that the state is made of DeviceArrays, this is false for the non-jitted.
    assert_is_jax_array_tree(state)

    # Call again to check it does not compile twice.
    state, timestep = reset_fn(key)
    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)


def test_solitaire__step_jit(solitaire: Solitaire) -> None:
    """Confirm that the step is only compiled once when jitted."""
    key = jax.random.PRNGKey(0)
    state, timestep = solitaire.reset(key)
    action = jnp.argwhere(state.action_mask)[0]

    chex.clear_trace_counter()
    step_fn = jax.jit(chex.assert_max_traces(solitaire.step, n=1))

    new_state, next_timestep = step_fn(state, action)
    # Check that the state has changed.
    assert not jnp.array_equal(new_state.board, state.board)
    assert new_state.step_count == state.step_count + 1
    assert new_state.remaining == state.remaining - 1
    assert jnp.sum(new_state.board) == jnp.sum(state.board) - 1

    # Check that the state is made of DeviceArrays, this is false for the non-jitted.
    assert_is_jax_array_tree(new_state)

    # New step
    state = new_state
    action = jnp.argwhere(state.action_mask)[0]
    new_state, next_timestep = step_fn(state, action)

    # Check that the state has changed
    assert not jnp.array_equal(new_state.board, state.board)
    assert new_state.step_count == state.step_count + 1
    assert new_state.remaining == state.remaining - 1
    assert jnp.sum(new_state.board) == jnp.sum(state.board) - 1


def test_solitaire__step_invalid(solitaire: Solitaire, board7x7: Board) -> None:
    """Confirm that performing step on an invalid action does nothing."""
    state = State(
        board=board7x7,
        step_count=jnp.array(0),
        action_mask=solitaire._get_action_mask(board7x7),
        remaining=jnp.sum(board7x7),
        key=jax.random.PRNGKey(0),
    )
    # Invalid action.
    action = (0, 0, 0)
    step_fn = jax.jit(solitaire.step)
    new_state, next_timestep = step_fn(state, action)
    assert jnp.array_equal(state.board, new_state.board)
    assert jnp.array_equal(state.step_count + 1, new_state.step_count)
    assert jnp.array_equal(state.action_mask, new_state.action_mask)
    assert jnp.array_equal(state.remaining, new_state.remaining)


def test_solitaire__step_position(
    solitaire: Solitaire, starting_board5x5: Board
) -> None:
    """Verify that the action mask returned from `step` is correct."""
    state = State(
        board=starting_board5x5,
        step_count=jnp.array(0),
        action_mask=solitaire._get_action_mask(starting_board5x5),
        remaining=jnp.sum(starting_board5x5),
        key=jax.random.PRNGKey(0),
    )
    # Move top, middle peg down.
    action = (0, 2, 2)
    step_fn = jax.jit(solitaire.step)
    new_state, next_timestep = step_fn(state, action)
    assert jnp.array_equal(
        new_state.board,
        [
            [0, 1, 0, 1, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
    )


def test_solitaire__generate_board(solitaire: Solitaire) -> None:
    """Confirm that `generate_board` method creates the correct board."""
    key = jax.random.PRNGKey(0)
    generate_board = jax.jit(solitaire._generate_board)
    board = generate_board(key)
    # Check that all the tiles are zero except one tile with value (1 or 2).
    assert jnp.array_equal(
        board,
        jnp.array(
            [
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
            ]
        ),
    )

    # Check the same board is used in reset.
    reset_fn = jax.jit(solitaire.reset)
    state, timestep = reset_fn(key)
    assert jnp.array_equal(state.board, board)


def test_solitaire__get_action_mask(
    solitaire: Solitaire, starting_board5x5: Board
) -> None:
    """Verify that the action mask generated by `get_action_mask` is correct."""
    action_mask_fn = jax.jit(solitaire._get_action_mask)
    action_mask = action_mask_fn(starting_board5x5)
    assert jnp.array_equal(
        action_mask,
        jnp.stack(
            (
                jnp.stack(
                    (
                        jnp.array(
                            [
                                [
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                ],
                                [
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                ],
                                [
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                ],
                                [
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                ],
                                [
                                    0,
                                    0,
                                    1,
                                    0,
                                    0,
                                ],
                            ]
                        ),
                        jnp.array(
                            [
                                [
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                ],
                                [
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                ],
                                [
                                    1,
                                    0,
                                    0,
                                    0,
                                    0,
                                ],
                                [
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                ],
                                [
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                ],
                            ]
                        ),
                        jnp.array(
                            [
                                [
                                    0,
                                    0,
                                    1,
                                    0,
                                    0,
                                ],
                                [
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                ],
                                [
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                ],
                                [
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                ],
                                [
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                ],
                            ]
                        ),
                        jnp.array(
                            [
                                [
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                ],
                                [
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                ],
                                [
                                    0,
                                    0,
                                    0,
                                    0,
                                    1,
                                ],
                                [
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                ],
                                [
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                ],
                            ]
                        ),
                    ),
                    axis=-1,
                )
            )
        ),
    )


def test_solitaire__does_not_smoke(solitaire: Solitaire) -> None:
    """Test that we can run an episode without any errors."""
    check_env_does_not_smoke(solitaire)
