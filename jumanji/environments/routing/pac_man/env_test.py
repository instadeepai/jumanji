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

from jumanji.environments.routing.pac_man.env import PacMan
from jumanji.environments.routing.pac_man.types import Position, State
from jumanji.testing.env_not_smoke import check_env_does_not_smoke
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import TimeStep


@pytest.fixture
def pac_man() -> PacMan:
    """Instantiates a default pacman environment."""
    return PacMan()


def test_pac_man__reset_jit(pac_man: PacMan) -> None:
    """Confirm that the reset method is only compiled once when jitted."""
    chex.clear_trace_counter()
    reset_fn = jax.jit(chex.assert_max_traces(pac_man.reset, n=1))
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


def test_pac_man__step_jit(pac_man: PacMan) -> None:
    """Confirm that the step is only compiled once when jitted."""
    key = jax.random.PRNGKey(0)
    state, timestep = pac_man.reset(key)
    action = 3

    chex.clear_trace_counter()
    step_fn = jax.jit(chex.assert_max_traces(pac_man.step, n=1))

    new_state, next_timestep = step_fn(state, action)
    # Check that the playerposition has changed
    assert jnp.array_equal(new_state.player_locations.x, state.player_locations.x)
    assert not jnp.array_equal(new_state.player_locations.y, state.player_locations.y)
    assert jnp.array_equal(new_state.ghost_locations, state.ghost_locations)
    assert not jnp.array_equal(new_state.pellets, state.pellets)  #
    assert not jnp.array_equal(new_state.ghost_starts, state.ghost_starts)

    # Check that the state is made of DeviceArrays, this is false for the non-jitted.
    assert_is_jax_array_tree(new_state)

    # New step
    state = new_state
    action = 3
    new_state, next_timestep = step_fn(state, action)

    # Check that the state has changed
    assert new_state.player_locations.x == state.player_locations.x
    assert new_state.player_locations.y != state.player_locations.y
    assert jnp.array_equal(new_state.ghost_locations, state.ghost_locations)
    assert new_state.pellets != state.pellets


def test_pac_man_step_invalid(pac_man: PacMan) -> None:
    key = jax.random.PRNGKey(0)
    state, timestep = pac_man.reset(key)
    action = 2

    step_fn = jax.jit(pac_man.step)
    new_state, next_timestep = step_fn(state, action)

    assert new_state.player_locations.y == state.player_locations.y
    assert new_state.player_locations.y == state.player_locations.y


def test_pac_man_does_not_smoke(pac_man: PacMan) -> None:

    check_env_does_not_smoke(pac_man)


def test_power_pellet(pac_man: PacMan) -> None:

    key = jax.random.PRNGKey(0)
    state, timestep = pac_man.reset(key)

    # move player and ghost to pellet
    state.player_locations = Position(x=2, y=6)
    state.ghost_locations = jnp.array([[7, 4], [14, 13], [13, 14], [14, 14]])
    state.ghost_init_steps = jnp.array([5, 10, 15, 20])

    step_fn = jax.jit(pac_man.step)
    action = 2
    new_state, next_timestep = step_fn(state, action)

    assert not jnp.array_equal(new_state.power_up_locations, state.power_up_locations)
    assert new_state.frightened_state_time != state.frightened_state_time
