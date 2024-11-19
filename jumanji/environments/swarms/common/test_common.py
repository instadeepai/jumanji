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

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import pytest

from jumanji.environments.swarms.common import types, updates, viewer


@pytest.fixture
def params() -> types.AgentParams:
    return types.AgentParams(
        max_rotate=0.5,
        max_accelerate=0.01,
        min_speed=0.01,
        max_speed=0.05,
        view_angle=0.5,
    )


@pytest.mark.parametrize(
    "heading, speed, actions, expected",
    [
        [0.0, 0.01, [1.0, 0.0], (0.5 * jnp.pi, 0.01)],
        [0.0, 0.01, [-1.0, 0.0], (1.5 * jnp.pi, 0.01)],
        [jnp.pi, 0.01, [1.0, 0.0], (1.5 * jnp.pi, 0.01)],
        [jnp.pi, 0.01, [-1.0, 0.0], (0.5 * jnp.pi, 0.01)],
        [1.75 * jnp.pi, 0.01, [1.0, 0.0], (0.25 * jnp.pi, 0.01)],
        [0.0, 0.01, [0.0, 1.0], (0.0, 0.02)],
        [0.0, 0.01, [0.0, -1.0], (0.0, 0.01)],
        [0.0, 0.02, [0.0, -1.0], (0.0, 0.01)],
        [0.0, 0.05, [0.0, -1.0], (0.0, 0.04)],
        [0.0, 0.05, [0.0, 1.0], (0.0, 0.05)],
    ],
)
def test_velocity_update(
    params: types.AgentParams,
    heading: float,
    speed: float,
    actions: List[float],
    expected: Tuple[float, float],
) -> None:
    key = jax.random.PRNGKey(101)

    state = types.AgentState(
        pos=jnp.zeros((1, 2)),
        heading=jnp.array([heading]),
        speed=jnp.array([speed]),
    )
    actions = jnp.array([actions])

    new_heading, new_speed = updates.update_velocity(key, params, (actions, state))

    assert jnp.isclose(new_heading[0], expected[0])
    assert jnp.isclose(new_speed[0], expected[1])


@pytest.mark.parametrize(
    "pos, heading, speed, expected, env_size",
    [
        [[0.0, 0.5], 0.0, 0.1, [0.1, 0.5], 1.0],
        [[0.0, 0.5], jnp.pi, 0.1, [0.9, 0.5], 1.0],
        [[0.5, 0.0], 0.5 * jnp.pi, 0.1, [0.5, 0.1], 1.0],
        [[0.5, 0.0], 1.5 * jnp.pi, 0.1, [0.5, 0.9], 1.0],
        [[0.4, 0.2], 0.0, 0.2, [0.1, 0.2], 0.5],
        [[0.1, 0.2], jnp.pi, 0.2, [0.4, 0.2], 0.5],
        [[0.2, 0.4], 0.5 * jnp.pi, 0.2, [0.2, 0.1], 0.5],
        [[0.2, 0.1], 1.5 * jnp.pi, 0.2, [0.2, 0.4], 0.5],
    ],
)
def test_move(
    pos: List[float], heading: float, speed: float, expected: List[float], env_size: float
) -> None:
    pos = jnp.array(pos)
    new_pos = updates.move(pos, heading, speed, env_size)

    assert jnp.allclose(new_pos, jnp.array(expected))


@pytest.mark.parametrize(
    "pos, heading, speed, actions, expected_pos, expected_heading, expected_speed, env_size",
    [
        [[0.0, 0.5], 0.0, 0.01, [0.0, 0.0], [0.01, 0.5], 0.0, 0.01, 1.0],
        [[0.5, 0.0], 0.0, 0.01, [1.0, 0.0], [0.5, 0.01], 0.5 * jnp.pi, 0.01, 1.0],
        [[0.5, 0.0], 0.0, 0.01, [-1.0, 0.0], [0.5, 0.99], 1.5 * jnp.pi, 0.01, 1.0],
        [[0.0, 0.5], 0.0, 0.01, [0.0, 1.0], [0.02, 0.5], 0.0, 0.02, 1.0],
        [[0.0, 0.5], 0.0, 0.01, [0.0, -1.0], [0.01, 0.5], 0.0, 0.01, 1.0],
        [[0.0, 0.5], 0.0, 0.05, [0.0, 1.0], [0.05, 0.5], 0.0, 0.05, 1.0],
        [[0.495, 0.25], 0.0, 0.01, [0.0, 0.0], [0.005, 0.25], 0.0, 0.01, 0.5],
        [[0.25, 0.005], 1.5 * jnp.pi, 0.01, [0.0, 0.0], [0.25, 0.495], 1.5 * jnp.pi, 0.01, 0.5],
    ],
)
def test_state_update(
    params: types.AgentParams,
    pos: List[float],
    heading: float,
    speed: float,
    actions: List[float],
    expected_pos: List[float],
    expected_heading: float,
    expected_speed: float,
    env_size: float,
) -> None:
    key = jax.random.PRNGKey(101)

    state = types.AgentState(
        pos=jnp.array([pos]),
        heading=jnp.array([heading]),
        speed=jnp.array([speed]),
    )
    actions = jnp.array([actions])

    new_state = updates.update_state(key, env_size, params, state, actions)

    assert isinstance(new_state, types.AgentState)
    assert jnp.allclose(new_state.pos, jnp.array([expected_pos]))
    assert jnp.allclose(new_state.heading, jnp.array([expected_heading]))
    assert jnp.allclose(new_state.speed, jnp.array([expected_speed]))


def test_view_reduction() -> None:
    view_a = jnp.array([-1.0, -1.0, 0.2, 0.2, 0.5])
    view_b = jnp.array([-1.0, 0.2, -1.0, 0.5, 0.2])
    result = updates.view_reduction(view_a, view_b)
    assert jnp.allclose(result, jnp.array([-1.0, 0.2, 0.2, 0.2, 0.2]))


@pytest.mark.parametrize(
    "pos, view_angle, env_size, expected",
    [
        [[0.05, 0.0], 0.5, 1.0, [-1.0, -1.0, 0.5, -1.0, -1.0]],
        [[0.0, 0.05], 0.5, 1.0, [0.5, -1.0, -1.0, -1.0, -1.0]],
        [[0.0, 0.95], 0.5, 1.0, [-1.0, -1.0, -1.0, -1.0, 0.5]],
        [[0.95, 0.0], 0.5, 1.0, [-1.0, -1.0, -1.0, -1.0, -1.0]],
        [[0.05, 0.0], 0.25, 1.0, [-1.0, -1.0, 0.5, -1.0, -1.0]],
        [[0.0, 0.05], 0.25, 1.0, [-1.0, -1.0, -1.0, -1.0, -1.0]],
        [[0.0, 0.95], 0.25, 1.0, [-1.0, -1.0, -1.0, -1.0, -1.0]],
        [[0.01, 0.0], 0.5, 1.0, [-1.0, -1.0, 0.1, -1.0, -1.0]],
        [[0.0, 0.45], 0.5, 1.0, [4.5, -1.0, -1.0, -1.0, -1.0]],
        [[0.0, 0.45], 0.5, 0.5, [-1.0, -1.0, -1.0, -1.0, 0.5]],
    ],
)
def test_view(pos: List[float], view_angle: float, env_size: float, expected: List[float]) -> None:
    state_a = types.AgentState(
        pos=jnp.zeros((2,)),
        heading=0.0,
        speed=0.0,
    )

    state_b = types.AgentState(
        pos=jnp.array(pos),
        heading=0.0,
        speed=0.0,
    )

    obs = updates.view(
        None, (view_angle, 0.02), state_a, state_b, n_view=5, i_range=0.1, env_size=env_size
    )
    assert jnp.allclose(obs, jnp.array(expected))


def test_viewer_utils() -> None:
    f, ax = plt.subplots()
    f, ax = viewer.format_plot(f, ax, (1.0, 1.0))

    assert isinstance(f, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)

    state = types.AgentState(
        pos=jnp.zeros((3, 2)),
        heading=jnp.zeros((3,)),
        speed=jnp.zeros((3,)),
    )

    quiver = viewer.draw_agents(ax, state, "red")

    assert isinstance(quiver, matplotlib.quiver.Quiver)
