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

import jax
import jax.numpy as jnp
import pytest

from jumanji.environments.swarms.common.types import AgentParams, AgentState
from jumanji.environments.swarms.search_and_rescue.dynamics import RandomWalk, TargetDynamics
from jumanji.environments.swarms.search_and_rescue.generator import Generator, RandomGenerator
from jumanji.environments.swarms.search_and_rescue.types import State, TargetState
from jumanji.environments.swarms.search_and_rescue.utils import has_been_found, has_found_target


def test_random_generator() -> None:
    key = jax.random.PRNGKey(101)
    params = AgentParams(
        max_rotate=0.5,
        max_accelerate=0.01,
        min_speed=0.01,
        max_speed=0.05,
        view_angle=0.5,
    )
    generator = RandomGenerator(num_searchers=100, num_targets=101)

    assert isinstance(generator, Generator)

    state = generator(key, params)

    assert isinstance(state, State)
    assert state.searchers.pos.shape == (generator.num_searchers, 2)
    assert jnp.all(0.0 <= state.searchers.pos) and jnp.all(state.searchers.pos <= 1.0)
    assert state.targets.pos.shape == (generator.num_targets, 2)
    assert jnp.all(0.0 <= state.targets.pos) and jnp.all(state.targets.pos <= 1.0)
    assert not jnp.any(state.targets.found)
    assert state.step == 0


def test_random_walk_dynamics() -> None:
    n_targets = 50
    key = jax.random.PRNGKey(101)
    s0 = jnp.full((n_targets, 2), 0.5)

    dynamics = RandomWalk(0.1)
    assert isinstance(dynamics, TargetDynamics)
    s1 = dynamics(key, s0)

    assert s1.shape == (n_targets, 2)
    assert jnp.all(jnp.abs(s0 - s1) < 0.1)


@pytest.mark.parametrize(
    "pos, heading, view_angle, target_state, expected",
    [
        ([0.1, 0.0], 0.0, 0.5, False, False),
        ([0.1, 0.0], jnp.pi, 0.5, False, True),
        ([0.1, 0.0], jnp.pi, 0.5, True, True),
        ([0.9, 0.0], jnp.pi, 0.5, False, False),
        ([0.9, 0.0], 0.0, 0.5, False, True),
        ([0.9, 0.0], 0.0, 0.5, True, True),
        ([0.0, 0.1], 1.5 * jnp.pi, 0.5, True, True),
        ([0.1, 0.0], 0.5 * jnp.pi, 0.5, False, True),
        ([0.1, 0.0], 0.5 * jnp.pi, 0.4, False, False),
    ],
)
def test_target_found(
    pos: List[float],
    heading: float,
    view_angle: float,
    target_state: bool,
    expected: bool,
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

    found = has_been_found(None, view_angle, target.pos, searcher)
    reward = has_found_target(None, view_angle, searcher, target)

    assert found == expected

    if found and target_state:
        assert reward == 0.0
    elif found and not target_state:
        assert reward == 1.0
    elif not found:
        assert reward == 0.0
