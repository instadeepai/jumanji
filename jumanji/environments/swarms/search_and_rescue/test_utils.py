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

from typing import List

import chex
import jax.numpy as jnp
import pytest

from jumanji.environments.swarms.common.types import AgentState
from jumanji.environments.swarms.search_and_rescue.dynamics import RandomWalk, TargetDynamics
from jumanji.environments.swarms.search_and_rescue.types import TargetState
from jumanji.environments.swarms.search_and_rescue.utils import (
    reward_if_found_target,
    target_has_been_found,
)


def test_random_walk_dynamics(key: chex.PRNGKey) -> None:
    n_targets = 50
    s0 = jnp.full((n_targets, 2), 0.5)

    dynamics = RandomWalk(0.1)
    assert isinstance(dynamics, TargetDynamics)
    s1 = dynamics(key, s0)

    assert s1.shape == (n_targets, 2)
    assert jnp.all(jnp.abs(s0 - s1) < 0.1)


@pytest.mark.parametrize(
    "pos, heading, view_angle, target_state, expected, env_size",
    [
        ([0.1, 0.0], 0.0, 0.5, False, False, 1.0),
        ([0.1, 0.0], jnp.pi, 0.5, False, True, 1.0),
        ([0.1, 0.0], jnp.pi, 0.5, True, True, 1.0),
        ([0.9, 0.0], jnp.pi, 0.5, False, False, 1.0),
        ([0.9, 0.0], 0.0, 0.5, False, True, 1.0),
        ([0.9, 0.0], 0.0, 0.5, True, True, 1.0),
        ([0.0, 0.1], 1.5 * jnp.pi, 0.5, True, True, 1.0),
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
        found=target_state,
    )

    searcher = AgentState(
        pos=jnp.array(pos),
        heading=heading,
        speed=0.0,
    )

    found = target_has_been_found(None, view_angle, target.pos, searcher, env_size=env_size)
    reward = reward_if_found_target(None, view_angle, searcher, target, env_size=env_size)

    assert found == expected

    if found and target_state:
        assert reward == 0.0
    elif found and not target_state:
        assert reward == 1.0
    elif not found:
        assert reward == 0.0
