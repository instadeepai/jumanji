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

from functools import partial

import chex
import jax
import jax.numpy as jnp
from dm_env import StepType

from jumanji.environments.routing.connector.constants import EMPTY
from jumanji.environments.routing.connector.env import Connector
from jumanji.environments.routing.connector.types import Agent, State
from jumanji.environments.routing.connector.utils import get_position, get_target
from jumanji.types import TimeStep


@partial(jax.vmap, in_axes=(0, None))
def is_head_on_grid(agent: Agent, grid: chex.Array) -> chex.Array:
    """Returns true if the agent's head is on the correct place on the grid."""
    return (grid[agent.position] == get_position(agent.id)).any()


@partial(jax.vmap, in_axes=(0, None))
def is_target_on_grid(agent: Agent, grid: chex.Array) -> chex.Array:
    """Returns true if the agent's target is on the correct place on the grid."""
    return (grid[agent.target] == get_target(agent.id)).any()


def test_connector__reset(env: Connector, key: jax.random.KeyArray) -> None:
    """Test that all heads and targets are on the board."""
    reset = env.reset  # jax.jit(env.reset)
    state, timestep = reset(key)

    assert state.grid.shape == (env._size, env._size)

    for agent_id in range(env._num_agents):
        assert (state.grid == get_position(agent_id)).any()
        assert (state.grid == get_target(agent_id)).any()

    assert all(is_head_on_grid(state.agents, state.grid))
    assert all(is_target_on_grid(state.agents, state.grid))

    assert (timestep.discount == jnp.ones(env._num_agents, int)).all()
    assert (timestep.reward == jnp.zeros(env._num_agents)).all()
    assert timestep.step_type == StepType.FIRST


def test_connector__reset_jit(env: Connector) -> None:
    """Confirm that the reset is only compiled once when jitted."""
    chex.clear_trace_counter()
    reset_fn = jax.jit(chex.assert_max_traces(env.reset, n=1))
    key = jax.random.PRNGKey(0)
    state, timestep = reset_fn(key)

    # Call again to check it does not compile twice
    state, timestep = reset_fn(key)
    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)


def test_connector__obs_from_grid(
    env: Connector,
    grid: chex.Array,
    path0: int,
    path1: int,
    path2: int,
    targ0: int,
    targ1: int,
    targ2: int,
    posi0: int,
    posi1: int,
    posi2: int,
) -> None:
    """Tests that observations are correctly generated given the grid."""
    observations = env._obs_from_grid(grid)

    expected_agent_1 = jnp.array(
        [
            [EMPTY, EMPTY, targ2, EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, posi2, path2, path2, EMPTY],
            [EMPTY, EMPTY, EMPTY, targ1, posi1, EMPTY],
            [targ0, EMPTY, posi0, EMPTY, path1, EMPTY],
            [EMPTY, EMPTY, path0, EMPTY, path1, EMPTY],
            [EMPTY, EMPTY, path0, EMPTY, EMPTY, EMPTY],
        ]
    )
    expected_agent_2 = jnp.array(
        [
            [EMPTY, EMPTY, targ1, EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, posi1, path1, path1, EMPTY],
            [EMPTY, EMPTY, EMPTY, targ0, posi0, EMPTY],
            [targ2, EMPTY, posi2, EMPTY, path0, EMPTY],
            [EMPTY, EMPTY, path2, EMPTY, path0, EMPTY],
            [EMPTY, EMPTY, path2, EMPTY, EMPTY, EMPTY],
        ]
    )

    expected_obs = jnp.stack([grid, expected_agent_1, expected_agent_2])
    assert (expected_obs == observations).all()


def test_connector__get_action_mask(state: State, env: Connector) -> None:
    """Validates the action masking."""
    action_masks = jax.vmap(env._get_action_mask, (0, None))(state.agents, state.grid)
    expected_mask = jnp.array(
        [
            [True, True, False, True, True],
            [True, True, True, False, True],
            [True, False, True, False, True],
        ]
    )
    assert (action_masks == expected_mask).all()
