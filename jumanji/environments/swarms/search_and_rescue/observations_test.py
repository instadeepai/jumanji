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

from typing import List, Tuple

import chex
import jax.numpy as jnp
import pytest

from jumanji.environments.swarms.common.types import AgentState
from jumanji.environments.swarms.search_and_rescue import SearchAndRescue, observations
from jumanji.environments.swarms.search_and_rescue.types import State, TargetState

VISION_RANGE = 0.2
VIEW_ANGLE = 0.5


@pytest.mark.parametrize(
    "searcher_positions, searcher_headings, env_size, view_updates",
    [
        # Both out of view range
        ([[0.8, 0.5], [0.2, 0.5]], [jnp.pi, 0.0], 1.0, []),
        # Both view each other
        ([[0.25, 0.5], [0.2, 0.5]], [jnp.pi, 0.0], 1.0, [(0, 5, 0.25), (1, 5, 0.25)]),
        # One facing wrong direction
        (
            [[0.25, 0.5], [0.2, 0.5]],
            [jnp.pi, jnp.pi],
            1.0,
            [(0, 5, 0.25)],
        ),
        # Only see closest neighbour
        (
            [[0.35, 0.5], [0.25, 0.5], [0.2, 0.5]],
            [jnp.pi, 0.0, 0.0],
            1.0,
            [(0, 5, 0.5), (1, 5, 0.5), (2, 5, 0.25)],
        ),
        # Observed around wrapped edge
        (
            [[0.025, 0.5], [0.975, 0.5]],
            [jnp.pi, 0.0],
            1.0,
            [(0, 5, 0.25), (1, 5, 0.25)],
        ),
        # Observed around wrapped edge of smaller env
        (
            [[0.025, 0.25], [0.475, 0.25]],
            [jnp.pi, 0.0],
            0.5,
            [(0, 5, 0.25), (1, 5, 0.25)],
        ),
    ],
)
def test_searcher_view(
    key: chex.PRNGKey,
    # env: SearchAndRescue,
    searcher_positions: List[List[float]],
    searcher_headings: List[float],
    env_size: float,
    view_updates: List[Tuple[int, int, float]],
) -> None:
    """
    Test agent-only view model generates expected array with different
    configurations of agents.
    """

    searcher_positions = jnp.array(searcher_positions)
    searcher_headings = jnp.array(searcher_headings)
    searcher_speed = jnp.zeros(searcher_headings.shape)

    state = State(
        searchers=AgentState(
            pos=searcher_positions, heading=searcher_headings, speed=searcher_speed
        ),
        targets=TargetState(
            pos=jnp.zeros((1, 2)), vel=jnp.zeros((1, 2)), found=jnp.zeros((1, 2), dtype=bool)
        ),
        key=key,
    )

    observe_fn = observations.AgentObservationFn(
        num_vision=11,
        vision_range=VISION_RANGE,
        view_angle=VIEW_ANGLE,
        agent_radius=0.01,
        env_size=env_size,
    )

    obs = observe_fn(state)

    expected = jnp.full((searcher_headings.shape[0], 1, observe_fn.num_vision), -1.0)

    for i, idx, val in view_updates:
        expected = expected.at[i, 0, idx].set(val)

    assert jnp.all(jnp.isclose(obs, expected))


@pytest.mark.parametrize(
    "searcher_positions, searcher_headings, env_size, view_updates",
    [
        # Both out of view range
        ([[0.8, 0.5], [0.2, 0.5]], [jnp.pi, 0.0], 1.0, []),
        # Both view each other
        ([[0.25, 0.5], [0.2, 0.5]], [jnp.pi, 0.0], 1.0, [(0, 5, 0.25), (1, 5, 0.25)]),
        # One facing wrong direction
        (
            [[0.25, 0.5], [0.2, 0.5]],
            [jnp.pi, jnp.pi],
            1.0,
            [(0, 5, 0.25)],
        ),
        # Only see closest neighbour
        (
            [[0.35, 0.5], [0.25, 0.5], [0.2, 0.5]],
            [jnp.pi, 0.0, 0.0],
            1.0,
            [(0, 5, 0.5), (1, 5, 0.5), (2, 5, 0.25)],
        ),
        # Observed around wrapped edge
        (
            [[0.025, 0.5], [0.975, 0.5]],
            [jnp.pi, 0.0],
            1.0,
            [(0, 5, 0.25), (1, 5, 0.25)],
        ),
        # Observed around wrapped edge of smaller env
        (
            [[0.025, 0.25], [0.475, 0.25]],
            [jnp.pi, 0.0],
            0.5,
            [(0, 5, 0.25), (1, 5, 0.25)],
        ),
    ],
)
def test_search_and_target_view_searchers(
    key: chex.PRNGKey,
    searcher_positions: List[List[float]],
    searcher_headings: List[float],
    env_size: float,
    view_updates: List[Tuple[int, int, float]],
) -> None:
    """
    Test agent+target view model generates expected array with different
    configurations of agents only.
    """

    n_agents = len(searcher_headings)
    searcher_positions = jnp.array(searcher_positions)
    searcher_headings = jnp.array(searcher_headings)
    searcher_speed = jnp.zeros(searcher_headings.shape)

    state = State(
        searchers=AgentState(
            pos=searcher_positions, heading=searcher_headings, speed=searcher_speed
        ),
        targets=TargetState(
            pos=jnp.zeros((1, 2)), vel=jnp.zeros((1, 2)), found=jnp.zeros((1,), dtype=bool)
        ),
        key=key,
    )

    observe_fn = observations.AgentAndTargetObservationFn(
        num_vision=11,
        vision_range=VISION_RANGE,
        view_angle=VIEW_ANGLE,
        agent_radius=0.01,
        env_size=env_size,
    )

    obs = observe_fn(state)
    assert obs.shape == (n_agents, 2, observe_fn.num_vision)

    expected = jnp.full((n_agents, 2, observe_fn.num_vision), -1.0)

    for i, idx, val in view_updates:
        expected = expected.at[i, 0, idx].set(val)

    assert jnp.all(jnp.isclose(obs, expected))


@pytest.mark.parametrize(
    "searcher_position, searcher_heading, target_position, target_found, env_size, view_updates",
    [
        # Target out of view range
        ([0.8, 0.5], jnp.pi, [0.2, 0.5], True, 1.0, []),
        # Target in view and found
        ([0.25, 0.5], jnp.pi, [0.2, 0.5], True, 1.0, [(5, 0.25)]),
        # Target in view but not found
        ([0.25, 0.5], jnp.pi, [0.2, 0.5], False, 1.0, []),
        # Observed around wrapped edge
        (
            [0.025, 0.5],
            jnp.pi,
            [0.975, 0.5],
            True,
            1.0,
            [(5, 0.25)],
        ),
        # Observed around wrapped edge of smaller env
        (
            [0.025, 0.25],
            jnp.pi,
            [0.475, 0.25],
            True,
            0.5,
            [(5, 0.25)],
        ),
    ],
)
def test_search_and_target_view_targets(
    key: chex.PRNGKey,
    env: SearchAndRescue,
    searcher_position: List[float],
    searcher_heading: float,
    target_position: List[float],
    target_found: bool,
    env_size: float,
    view_updates: List[Tuple[int, float]],
) -> None:
    """
    Test agent+target view model generates expected array with different
    configurations of targets only.
    """

    searcher_position = jnp.array([searcher_position])
    searcher_heading = jnp.array([searcher_heading])
    searcher_speed = jnp.zeros((1,))
    target_position = jnp.array([target_position])
    target_found = jnp.array([target_found])

    state = State(
        searchers=AgentState(pos=searcher_position, heading=searcher_heading, speed=searcher_speed),
        targets=TargetState(
            pos=target_position,
            vel=jnp.zeros_like(target_position),
            found=target_found,
        ),
        key=key,
    )

    observe_fn = observations.AgentAndTargetObservationFn(
        num_vision=11,
        vision_range=VISION_RANGE,
        view_angle=VIEW_ANGLE,
        agent_radius=0.01,
        env_size=env_size,
    )

    obs = observe_fn(state)
    assert obs.shape == (1, 2, observe_fn.num_vision)

    expected = jnp.full((1, 2, observe_fn.num_vision), -1.0)

    for idx, val in view_updates:
        expected = expected.at[0, 1, idx].set(val)

    assert jnp.all(jnp.isclose(obs, expected))


@pytest.mark.parametrize(
    "searcher_position, searcher_heading, target_position, target_found, env_size, view_updates",
    [
        # Target out of view range
        ([0.8, 0.5], jnp.pi, [0.2, 0.5], True, 1.0, []),
        # Target in view and found
        ([0.25, 0.5], jnp.pi, [0.2, 0.5], True, 1.0, [(1, 5, 0.25)]),
        # Target in view but not found
        ([0.25, 0.5], jnp.pi, [0.2, 0.5], False, 1.0, [(2, 5, 0.25)]),
        # Observed around wrapped edge found
        (
            [0.025, 0.5],
            jnp.pi,
            [0.975, 0.5],
            True,
            1.0,
            [(1, 5, 0.25)],
        ),
        # Observed around wrapped edge not found
        (
            [0.025, 0.5],
            jnp.pi,
            [0.975, 0.5],
            False,
            1.0,
            [(2, 5, 0.25)],
        ),
        # Observed around wrapped edge of smaller env
        (
            [0.025, 0.25],
            jnp.pi,
            [0.475, 0.25],
            True,
            0.5,
            [(1, 5, 0.25)],
        ),
    ],
)
def test_search_and_all_target_view_targets(
    key: chex.PRNGKey,
    env: SearchAndRescue,
    searcher_position: List[float],
    searcher_heading: float,
    target_position: List[float],
    target_found: bool,
    env_size: float,
    view_updates: List[Tuple[int, int, float]],
) -> None:
    """
    Test agent+target view model generates expected array with different
    configurations of targets only.
    """

    searcher_position = jnp.array([searcher_position])
    searcher_heading = jnp.array([searcher_heading])
    searcher_speed = jnp.zeros((1,))
    target_position = jnp.array([target_position])
    target_found = jnp.array([target_found])

    state = State(
        searchers=AgentState(pos=searcher_position, heading=searcher_heading, speed=searcher_speed),
        targets=TargetState(
            pos=target_position,
            vel=jnp.zeros_like(target_position),
            found=target_found,
        ),
        key=key,
    )

    observe_fn = observations.AgentAndAllTargetObservationFn(
        num_vision=11,
        vision_range=VISION_RANGE,
        view_angle=VIEW_ANGLE,
        agent_radius=0.01,
        env_size=env_size,
    )

    obs = observe_fn(state)
    assert obs.shape == (1, 3, observe_fn.num_vision)

    expected = jnp.full((1, 3, observe_fn.num_vision), -1.0)

    for i, idx, val in view_updates:
        expected = expected.at[0, i, idx].set(val)

    assert jnp.all(jnp.isclose(obs, expected))
