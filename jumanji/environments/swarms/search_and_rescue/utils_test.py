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
from typing import List

import chex
import jax
import jax.numpy as jnp
import pytest

from jumanji.environments.swarms.common.types import AgentState
from jumanji.environments.swarms.search_and_rescue.dynamics import RandomWalk, TargetDynamics
from jumanji.environments.swarms.search_and_rescue.types import TargetState
from jumanji.environments.swarms.search_and_rescue.utils import (
    searcher_detect_targets,
)


def test_random_walk_dynamics(key: chex.PRNGKey) -> None:
    n_targets = 50
    pos_0 = jnp.full((n_targets, 2), 0.5)

    s0 = TargetState(
        pos=pos_0, vel=jnp.zeros((n_targets, 2)), found=jnp.zeros((n_targets,), dtype=bool)
    )

    dynamics = RandomWalk(0.1)
    assert isinstance(dynamics, TargetDynamics)
    s1 = dynamics(key, s0, 1.0)

    assert isinstance(s1, TargetState)
    assert s1.pos.shape == (n_targets, 2)
    assert jnp.array_equal(s0.found, s1.found)
    assert jnp.all(jnp.abs(s0.pos - s1.pos) < 0.1)


@pytest.mark.parametrize(
    "pos, heading, view_angle, target_state, expected, env_size",
    [
        ([0.1, 0.0], 0.0, 0.5, False, False, 1.0),
        ([0.1, 0.0], jnp.pi, 0.5, False, True, 1.0),
        ([0.1, 0.0], jnp.pi, 0.5, True, False, 1.0),
        ([0.9, 0.0], jnp.pi, 0.5, False, False, 1.0),
        ([0.9, 0.0], 0.0, 0.5, False, True, 1.0),
        ([0.9, 0.0], 0.0, 0.5, True, False, 1.0),
        ([0.0, 0.1], 1.5 * jnp.pi, 0.5, True, False, 1.0),
        ([0.1, 0.0], 0.5 * jnp.pi, 0.5, False, True, 1.0),
        ([0.1, 0.0], 0.5 * jnp.pi, 0.4, False, False, 1.0),
        ([0.4, 0.0], 0.0, 0.5, False, False, 1.0),
        ([0.4, 0.0], 0.0, 0.5, False, True, 0.5),
    ],
)
def test_target_found(
    pos: List[float],
    heading: float,
    view_angle: float,
    target_state: bool,
    expected: bool,
    env_size: float,
) -> None:
    target = TargetState(
        pos=jnp.zeros((2,)),
        vel=jnp.zeros((2,)),
        found=target_state,
    )

    searcher = AgentState(
        pos=jnp.array(pos),
        heading=heading,
        speed=0.0,
    )

    found = jax.jit(partial(searcher_detect_targets, env_size=env_size, n_targets=1))(
        None,
        view_angle,
        searcher,
        (jnp.arange(1), target),
    )

    assert found.shape == (1,)
    assert found[0] == expected
