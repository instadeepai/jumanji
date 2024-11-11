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
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import py
import pytest

from jumanji.environments.swarms.common.types import AgentState
from jumanji.environments.swarms.search_and_rescue import SearchAndRescue
from jumanji.environments.swarms.search_and_rescue.dynamics import RandomWalk
from jumanji.environments.swarms.search_and_rescue.types import (
    Observation,
    State,
    TargetState,
)
from jumanji.testing.env_not_smoke import check_env_does_not_smoke, check_env_specs_does_not_smoke
from jumanji.types import StepType, TimeStep

SEARCHER_VISION_RANGE = 0.2
TARGET_CONTACT_RANGE = 0.05
AGENT_RADIUS = 0.05


@pytest.fixture
def env() -> SearchAndRescue:
    return SearchAndRescue(
        num_searchers=10,
        num_targets=20,
        searcher_vision_range=SEARCHER_VISION_RANGE,
        target_contact_range=TARGET_CONTACT_RANGE,
        num_vision=11,
        agent_radius=AGENT_RADIUS,
        searcher_max_rotate=0.2,
        searcher_max_accelerate=0.01,
        searcher_min_speed=0.01,
        searcher_max_speed=0.05,
        searcher_view_angle=0.5,
        max_steps=25,
    )


def test_env_init(env: SearchAndRescue) -> None:
    """
    Check newly initialised state has expected array shapes
    and initial timestep.
    """
    k = jax.random.PRNGKey(101)
    state, timestep = env.reset(k)
    assert isinstance(state, State)

    assert isinstance(state.searchers, AgentState)
    assert state.searchers.pos.shape == (env.num_searchers, 2)
    assert state.searchers.speed.shape == (env.num_searchers,)
    assert state.searchers.speed.shape == (env.num_searchers,)

    assert isinstance(state.targets, TargetState)
    assert state.targets.pos.shape == (env.num_targets, 2)
    assert state.targets.found.shape == (env.num_targets,)
    assert jnp.array_equal(state.targets.found, jnp.full((env.num_targets,), False, dtype=bool))
    assert state.step == 0

    assert isinstance(timestep.observation, Observation)
    assert timestep.observation.searcher_views.shape == (
        env.num_searchers,
        env.num_vision,
    )
    assert timestep.step_type == StepType.FIRST


def test_env_step(env: SearchAndRescue) -> None:
    """
    Run several steps of the environment with random actions and
    check states (i.e. positions, heading, speeds) all fall
    inside expected ranges.
    """
    key = jax.random.PRNGKey(101)
    n_steps = 22

    def step(
        carry: Tuple[chex.PRNGKey, State], _: None
    ) -> Tuple[Tuple[chex.PRNGKey, State], Tuple[State, TimeStep[Observation]]]:
        k, state = carry
        k, k_search = jax.random.split(k)
        actions = jax.random.uniform(k_search, (env.num_searchers, 2), minval=-1.0, maxval=1.0)
        new_state, timestep = env.step(state, actions)
        return (k, new_state), (state, timestep)

    init_state, _ = env.reset(key)
    (_, final_state), (state_history, timesteps) = jax.lax.scan(
        step, (key, init_state), length=n_steps
    )

    assert isinstance(state_history, State)

    assert state_history.searchers.pos.shape == (n_steps, env.num_searchers, 2)
    assert jnp.all((0.0 <= state_history.searchers.pos) & (state_history.searchers.pos <= 1.0))
    assert state_history.searchers.speed.shape == (n_steps, env.num_searchers)
    assert jnp.all(
        (env.searcher_params.min_speed <= state_history.searchers.speed)
        & (state_history.searchers.speed <= env.searcher_params.max_speed)
    )
    assert state_history.searchers.speed.shape == (n_steps, env.num_searchers)
    assert jnp.all(
        (0.0 <= state_history.searchers.heading) & (state_history.searchers.heading <= 2.0 * jnp.pi)
    )

    assert state_history.targets.pos.shape == (n_steps, env.num_targets, 2)
    assert jnp.all((0.0 <= state_history.targets.pos) & (state_history.targets.pos <= 1.0))


def test_env_does_not_smoke(env: SearchAndRescue) -> None:
    """Test that we can run an episode without any errors."""
    env.max_steps = 10

    def select_action(action_key: chex.PRNGKey, _state: Observation) -> chex.Array:
        return jax.random.uniform(action_key, (env.num_searchers, 2), minval=-1.0, maxval=1.0)

    check_env_does_not_smoke(env, select_action=select_action)


def test_env_specs_do_not_smoke(env: SearchAndRescue) -> None:
    """Test that we can access specs without any errors."""
    check_env_specs_does_not_smoke(env)


@pytest.mark.parametrize(
    "searcher_positions, searcher_headings, view_updates",
    [
        # Both out of view range
        ([[0.8, 0.5], [0.2, 0.5]], [jnp.pi, 0.0], []),
        # Both view each other
        ([[0.25, 0.5], [0.2, 0.5]], [jnp.pi, 0.0], [(0, 5, 0.25), (1, 5, 0.25)]),
        # One facing wrong direction
        (
            [[0.25, 0.5], [0.2, 0.5]],
            [jnp.pi, jnp.pi],
            [(0, 5, 0.25)],
        ),
        # Only see closest neighbour
        (
            [[0.35, 0.5], [0.25, 0.5], [0.2, 0.5]],
            [jnp.pi, 0.0, 0.0],
            [(0, 5, 0.5), (1, 5, 0.5), (2, 5, 0.25)],
        ),
        # Observed around wrapped edge
        (
            [[0.025, 0.5], [0.975, 0.5]],
            [jnp.pi, 0.0],
            [(0, 5, 0.25), (1, 5, 0.25)],
        ),
    ],
)
def test_searcher_view(
    env: SearchAndRescue,
    searcher_positions: List[List[float]],
    searcher_headings: List[float],
    view_updates: List[Tuple[int, int, float]],
) -> None:
    """
    Test view model generates expected array with different
    configurations of agents.
    """

    searcher_positions = jnp.array(searcher_positions)
    searcher_headings = jnp.array(searcher_headings)
    searcher_speed = jnp.zeros(searcher_headings.shape)

    state = State(
        searchers=AgentState(
            pos=searcher_positions, heading=searcher_headings, speed=searcher_speed
        ),
        targets=TargetState(pos=jnp.zeros((1, 2)), found=jnp.zeros((1, 2), dtype=bool)),
        key=jax.random.PRNGKey(101),
    )

    obs = env._state_to_observation(state)

    assert isinstance(obs, Observation)

    expected = jnp.ones((searcher_headings.shape[0], env.num_vision))

    for i, idx, val in view_updates:
        expected = expected.at[i, idx].set(val)

    assert jnp.all(jnp.isclose(obs.searcher_views, expected))


def test_target_detection(env: SearchAndRescue) -> None:
    # Keep targets in one location
    env._target_dynamics = RandomWalk(step_size=0.0)

    # Agent facing wrong direction should not see target
    state = State(
        searchers=AgentState(
            pos=jnp.array([[0.5, 0.5]]), heading=jnp.array([jnp.pi]), speed=jnp.array([0.0])
        ),
        targets=TargetState(pos=jnp.array([[0.54, 0.5]]), found=jnp.array([False])),
        key=jax.random.PRNGKey(101),
    )
    state, timestep = env.step(state, jnp.zeros((1, 2)))
    assert not state.targets.found[0]
    assert timestep.reward[0] == 0

    # Rotated agent should detect target
    state = State(
        searchers=AgentState(
            pos=state.searchers.pos, heading=jnp.array([0.0]), speed=state.searchers.speed
        ),
        targets=state.targets,
        key=state.key,
    )
    state, timestep = env.step(state, jnp.zeros((1, 2)))
    assert state.targets.found[0]
    assert timestep.reward[0] == 1

    # Once detected should remain detected
    state = State(
        searchers=AgentState(
            pos=jnp.array([[0.0, 0.0]]),
            heading=state.searchers.heading,
            speed=state.searchers.speed,
        ),
        targets=state.targets,
        key=state.key,
    )
    state, timestep = env.step(state, jnp.zeros((1, 2)))
    assert state.targets.found[0]
    assert timestep.reward[0] == 0


def test_search_and_rescue_render(monkeypatch: pytest.MonkeyPatch, env: SearchAndRescue) -> None:
    """Check that the render method builds the figure but does not display it."""
    monkeypatch.setattr(plt, "show", lambda fig: None)
    step_fn = jax.jit(env.step)
    state, timestep = env.reset(jax.random.PRNGKey(0))
    action = env.action_spec.generate_value()
    state, timestep = step_fn(state, action)
    env.render(state)
    env.close()


def test_search_and_rescue__animation(env: SearchAndRescue, tmpdir: py.path.local) -> None:
    """Check that the animation method creates the animation correctly and can save to a gif."""
    step_fn = jax.jit(env.step)
    state, _ = env.reset(jax.random.PRNGKey(0))
    states = [state]
    action = env.action_spec.generate_value()
    state, _ = step_fn(state, action)
    states.append(state)
    animation = env.animate(states, interval=200, save_path=None)
    assert isinstance(animation, matplotlib.animation.Animation)

    path = str(tmpdir.join("/anim.gif"))
    animation.save(path, writer=matplotlib.animation.PillowWriter(fps=10), dpi=60)
