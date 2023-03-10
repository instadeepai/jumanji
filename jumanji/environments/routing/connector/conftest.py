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
from jumanji.environments.routing.connector.env import Connector
from jumanji.environments.routing.connector.types import Agent, State
from jumanji.environments.routing.connector.utils import (
    get_path,
    get_position,
    get_target,
)


@pytest.fixture
def key() -> chex.PRNGKey:
    """A determinstic key."""
    return jax.random.PRNGKey(1)


@pytest.fixture
def path0() -> chex.Numeric:
    "Returns: the path of agent 0."
    return get_path(0)


@pytest.fixture
def path1() -> chex.Numeric:
    "Returns: the path of agent 1."
    return get_path(1)


@pytest.fixture
def path2() -> chex.Numeric:
    "Returns: the path of agent 2."
    return get_path(2)


@pytest.fixture
def targ0() -> chex.Numeric:
    "Returns: the target of agent 0."
    return get_target(0)


@pytest.fixture
def targ1() -> chex.Numeric:
    "Returns: the target of agent 1."
    return get_target(1)


@pytest.fixture
def targ2() -> chex.Numeric:
    "Returns: the target of agent 2."
    return get_target(2)


@pytest.fixture
def posi0() -> chex.Numeric:
    "Returns: the position of agent 0."
    return get_position(0)


@pytest.fixture
def posi1() -> chex.Numeric:
    "Returns: the position of agent 1."
    return get_position(1)


@pytest.fixture
def posi2() -> chex.Numeric:
    "Returns: the position of agent 2."
    return get_position(2)


@pytest.fixture
def grid(
    path0: int,
    path1: int,
    path2: int,
    targ0: int,
    targ1: int,
    targ2: int,
    posi0: int,
    posi1: int,
    posi2: int,
) -> chex.Array:
    """A deterministic grid layout for the Connector environment."""
    return jnp.array(
        [
            [EMPTY, EMPTY, targ0, EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, posi0, path0, path0, EMPTY],
            [EMPTY, EMPTY, EMPTY, targ2, posi2, EMPTY],
            [targ1, EMPTY, posi1, EMPTY, path2, EMPTY],
            [EMPTY, EMPTY, path1, EMPTY, path2, EMPTY],
            [EMPTY, EMPTY, path1, EMPTY, EMPTY, EMPTY],
        ]
    )


@pytest.fixture
def state(key: chex.PRNGKey, grid: chex.Array) -> State:
    """Returns a determinstic state for the Connector environment"""
    agents = jax.vmap(Agent)(
        id=jnp.arange(3),
        start=jnp.array([(1, 4), (5, 2), (4, 4)]),
        target=jnp.array([(0, 2), (3, 0), (2, 3)]),
        position=jnp.array([(1, 2), (3, 2), (2, 4)]),
    )

    state = State(key=key, grid=grid, step=jnp.array(0, jnp.int32), agents=agents)

    return state


@pytest.fixture
def state1(
    key: chex.PRNGKey,
    path0: int,
    path1: int,
    path2: int,
    targ1: int,
    posi0: int,
    posi1: int,
    posi2: int,
) -> State:
    """Creates the state (with 3 agents) that results from taking the action [UP, LEFT, LEFT] in the
    state defined above. Results in agent 0 and 2 reaching their targets.
    """

    grid = jnp.array(
        [
            [EMPTY, EMPTY, posi0, EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, path0, path0, path0, EMPTY],
            [EMPTY, EMPTY, EMPTY, posi2, path2, EMPTY],
            [targ1, posi1, path1, EMPTY, path2, EMPTY],
            [EMPTY, EMPTY, path1, EMPTY, path2, EMPTY],
            [EMPTY, EMPTY, path1, EMPTY, EMPTY, EMPTY],
        ]
    )
    agents = jax.vmap(Agent)(
        id=jnp.arange(3),
        start=jnp.array([(1, 4), (5, 2), (4, 4)]),
        target=jnp.array([(0, 2), (3, 0), (2, 3)]),
        position=jnp.array([(0, 2), (3, 1), (2, 3)]),
    )

    return State(grid=grid, step=jnp.array(1, jnp.int32), agents=agents, key=key)


@pytest.fixture
def state2(
    key: chex.PRNGKey,
    path0: int,
    path1: int,
    path2: int,
    posi0: int,
    posi1: int,
    posi2: int,
) -> State:
    """Creates the state (with 3 agents) that results from taking the action [NOOP, LEFT, NOOP] in
    state1 defined in the fixture above. Leads to agent 1 reaching its target.
    """

    grid = jnp.array(
        [
            [EMPTY, EMPTY, posi0, EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, path0, path0, path0, EMPTY],
            [EMPTY, EMPTY, EMPTY, posi2, path2, EMPTY],
            [posi1, path1, path1, EMPTY, path2, EMPTY],
            [EMPTY, EMPTY, path1, EMPTY, path2, EMPTY],
            [EMPTY, EMPTY, path1, EMPTY, EMPTY, EMPTY],
        ]
    )

    agents = jax.vmap(Agent)(
        id=jnp.arange(3),
        start=jnp.array([(1, 4), (5, 2), (4, 4)]),
        target=jnp.array([(0, 2), (3, 0), (2, 3)]),
        position=jnp.array([(0, 2), (3, 0), (2, 3)]),
    )

    return State(grid=grid, step=jnp.array(2, jnp.int32), agents=agents, key=key)


@pytest.fixture
def action1() -> chex.Array:
    """Action to move from state to state1."""
    return jnp.array([UP, LEFT, LEFT])


@pytest.fixture
def action2() -> chex.Array:
    """Action to move from state1 to state2."""
    return jnp.array([NOOP, LEFT, NOOP])


@pytest.fixture
def env() -> Connector:
    """Returns a Connector environment of size 6 with 3 agents and a time_limit of 5."""
    return Connector(size=6, num_agents=3, time_limit=5)
