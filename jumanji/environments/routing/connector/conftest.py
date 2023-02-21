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

from jumanji.environments.routing.connector.constants import EMPTY
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
def grid() -> chex.Array:
    """A deterministic grid layout for the Connector environment."""
    path0 = get_path(0)
    path1 = get_path(1)
    path2 = get_path(2)

    targ0 = get_target(0)
    targ1 = get_target(1)
    targ2 = get_target(2)

    posi0 = get_position(0)
    posi1 = get_position(1)
    posi2 = get_position(2)

    empty = EMPTY

    return jnp.array(
        [
            [empty, empty, targ0, empty, empty, empty],
            [empty, empty, posi0, path0, path0, empty],
            [empty, empty, empty, targ2, posi2, empty],
            [targ1, empty, posi1, empty, path2, empty],
            [empty, empty, path1, empty, path2, empty],
            [empty, empty, path1, empty, empty, empty],
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

    state = State(key=key, grid=grid, step=0, agents=agents)

    return state


@pytest.fixture
def env() -> Connector:
    """Returns a Connector environment of size 6 with 3 agents."""
    return Connector(size=6, num_agents=3)
