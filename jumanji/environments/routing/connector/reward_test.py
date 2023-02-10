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

from jumanji.environments.routing.connector.constants import EMPTY, LEFT, NOOP, UP
from jumanji.environments.routing.connector.reward import DenseRewardFn, SparseRewardFn
from jumanji.environments.routing.connector.types import Agent, State
from jumanji.environments.routing.connector.utils import (
    get_path,
    get_position,
    get_target,
)


@pytest.fixture
def state1(key: chex.PRNGKey) -> State:
    """Creates the state (with 3 agents) that results from taking the action [UP, LEFT, LEFT] in the
    state defined in conftest.py. Results in agent 0 and 2 reaching their targets."""
    path0 = get_path(0)
    path1 = get_path(1)
    path2 = get_path(2)

    targ1 = get_target(1)

    posi0 = get_position(0)
    posi1 = get_position(1)
    posi2 = get_position(2)

    empty = EMPTY

    grid = jnp.array(
        [
            [empty, empty, posi0, empty, empty, empty],
            [empty, empty, path0, path0, path0, empty],
            [empty, empty, empty, posi2, path2, empty],
            [targ1, posi1, path1, empty, path2, empty],
            [empty, empty, path1, empty, path2, empty],
            [empty, empty, path1, empty, empty, empty],
        ]
    )
    agents = jax.vmap(Agent)(
        id=jnp.arange(3),
        start=jnp.array([(1, 4), (5, 2), (4, 4)]),
        target=jnp.array([(0, 2), (3, 0), (2, 3)]),
        position=jnp.array([(0, 2), (3, 1), (2, 3)]),
    )

    return State(grid=grid, step=1, agents=agents, key=key)


@pytest.fixture
def state2(key: chex.PRNGKey) -> State:
    """Creates the state (with 3 agents) that results from taking the action [NOOP, LEFT, NOOP] in
    state1 defined in the fixture above. Leads to agent 1 reaching its target."""
    path0 = get_path(0)
    path1 = get_path(1)
    path2 = get_path(2)

    posi0 = get_position(0)
    posi1 = get_position(1)
    posi2 = get_position(2)

    empty = EMPTY

    grid = jnp.array(
        [
            [empty, empty, posi0, empty, empty, empty],
            [empty, empty, path0, path0, path0, empty],
            [empty, empty, empty, posi2, path2, empty],
            [posi1, path1, path1, empty, path2, empty],
            [empty, empty, path1, empty, path2, empty],
            [empty, empty, path1, empty, empty, empty],
        ]
    )

    agents = jax.vmap(Agent)(
        id=jnp.arange(3),
        start=jnp.array([(1, 4), (5, 2), (4, 4)]),
        target=jnp.array([(0, 2), (3, 0), (2, 3)]),
        position=jnp.array([(0, 2), (3, 0), (2, 3)]),
    )

    return State(grid=grid, step=2, agents=agents, key=key)


@pytest.fixture
def action1() -> chex.Array:
    """Action to move to state1."""
    return jnp.array([UP, LEFT, LEFT])


@pytest.fixture
def action2() -> chex.Array:
    """Action to move to state2."""
    return jnp.array([NOOP, LEFT, NOOP])


def test_sparse_reward(
    state: State, state1: State, state2: State, action1: chex.Array, action2: chex.Array
) -> None:
    sparse_reward_fn = SparseRewardFn()
    # Reward of moving between the same states should be 0.
    reward = sparse_reward_fn(state, state, jnp.array([0, 0, 0]))
    assert (reward == jnp.zeros(3)).all()

    # Reward for no agents finished to some agents finished.
    reward = sparse_reward_fn(state, state1, action1)
    assert (reward == jnp.array([1.0, 0.0, 1.0])).all()

    # Reward for some agents finished to all agents finished.
    reward = sparse_reward_fn(state1, state2, action2)
    assert (reward == jnp.array([0.0, 1.0, 0.0])).all()

    # Reward of none finished to all finished.
    reward = sparse_reward_fn(state, state2, action1)
    assert (reward == jnp.ones(3)).all()

    # Reward of all finished to all finished.
    reward = sparse_reward_fn(state2, state2, jnp.zeros(3))
    assert (reward == jnp.zeros(3)).all()


def test_dense_reward(
    state: State, state1: State, state2: State, action1: chex.Array, action2: chex.Array
) -> None:
    timestep_reward = -0.03
    connected_reward = 0.1
    noop_reward = -0.01

    dense_rew_fn = DenseRewardFn(
        timestep_reward=timestep_reward,
        connected_reward=connected_reward,
        noop_reward=noop_reward,
    )

    # Reward of moving between the same states should be 0.
    reward = dense_rew_fn(state, state, jnp.array([0, 0, 0]))
    assert (reward == jnp.array([timestep_reward + noop_reward] * 3)).all()

    # Reward for no agents finished to some agents finished.
    reward = dense_rew_fn(state, state1, action1)
    assert (
        reward
        == jnp.array(
            [
                connected_reward + timestep_reward,
                timestep_reward,
                connected_reward + timestep_reward,
            ]
        )
    ).all()

    # Reward for some agents finished to all agents finished.
    reward = dense_rew_fn(state1, state2, action2)
    assert (reward == jnp.array([0.0, connected_reward + timestep_reward, 0.0])).all()

    # Reward of none finished to all finished, when all agents take a non-noop action.
    reward = dense_rew_fn(state, state2, action1)
    assert (reward == jnp.array([connected_reward + timestep_reward] * 3)).all()

    # Reward of all finished to all finished.
    reward = dense_rew_fn(state2, state2, jnp.zeros(3))
    assert (reward == jnp.zeros(3)).all()
