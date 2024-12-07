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
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import py
import pytest

from jumanji.environments.swarms.common.types import AgentParams, AgentState
from jumanji.environments.swarms.search_and_rescue import SearchAndRescue
from jumanji.environments.swarms.search_and_rescue.dynamics import RandomWalk
from jumanji.environments.swarms.search_and_rescue.types import (
    Observation,
    State,
    TargetState,
)
from jumanji.testing.env_not_smoke import check_env_does_not_smoke, check_env_specs_does_not_smoke
from jumanji.types import StepType, TimeStep


def test_env_init(env: SearchAndRescue, key: chex.PRNGKey) -> None:
    """
    Check newly initialised state has expected array shapes
    and initial timestep.
    """
    state, timestep = env.reset(key)
    assert isinstance(state, State)

    assert isinstance(state.searchers, AgentState)
    assert state.searchers.pos.shape == (env.generator.num_searchers, 2)
    assert state.searchers.speed.shape == (env.generator.num_searchers,)
    assert state.searchers.speed.shape == (env.generator.num_searchers,)

    assert isinstance(state.targets, TargetState)
    assert state.targets.pos.shape == (env.generator.num_targets, 2)
    assert state.targets.found.shape == (env.generator.num_targets,)
    assert jnp.array_equal(
        state.targets.found, jnp.full((env.generator.num_targets,), False, dtype=bool)
    )
    assert state.step == 0

    assert isinstance(timestep.observation, Observation)
    assert timestep.observation.searcher_views.shape == (
        env.generator.num_searchers,
        *env._observation.view_shape,
    )
    assert timestep.step_type == StepType.FIRST


@pytest.mark.parametrize("env_size", [1.0, 0.2])
def test_env_step(env: SearchAndRescue, key: chex.PRNGKey, env_size: float) -> None:
    """
    Run several steps of the environment with random actions and
    check states (i.e. positions, heading, speeds) all fall
    inside expected ranges.
    """
    n_steps = 22
    env.generator.env_size = env_size

    def step(
        carry: Tuple[chex.PRNGKey, State], _: None
    ) -> Tuple[Tuple[chex.PRNGKey, State], Tuple[State, TimeStep[Observation]]]:
        k, state = carry
        k, k_search = jax.random.split(k)
        actions = jax.random.uniform(
            k_search, (env.generator.num_searchers, 2), minval=-1.0, maxval=1.0
        )
        new_state, timestep = env.step(state, actions)
        return (k, new_state), (state, timestep)

    init_state, _ = env.reset(key)
    (_, final_state), (state_history, timesteps) = jax.lax.scan(
        step, (key, init_state), length=n_steps
    )

    assert isinstance(state_history, State)

    assert state_history.searchers.pos.shape == (n_steps, env.generator.num_searchers, 2)
    assert jnp.all((0.0 <= state_history.searchers.pos) & (state_history.searchers.pos <= env_size))
    assert state_history.searchers.speed.shape == (n_steps, env.generator.num_searchers)
    assert jnp.all(
        (env.searcher_params.min_speed <= state_history.searchers.speed)
        & (state_history.searchers.speed <= env.searcher_params.max_speed)
    )
    assert state_history.searchers.speed.shape == (n_steps, env.generator.num_searchers)
    assert jnp.all(
        (0.0 <= state_history.searchers.heading) & (state_history.searchers.heading <= 2.0 * jnp.pi)
    )

    assert state_history.targets.pos.shape == (n_steps, env.generator.num_targets, 2)
    assert jnp.all((0.0 <= state_history.targets.pos) & (state_history.targets.pos <= env_size))


def test_env_does_not_smoke(env: SearchAndRescue) -> None:
    """Test that we can run an episode without any errors."""
    env.max_steps = 10

    def select_action(action_key: chex.PRNGKey, _state: Observation) -> chex.Array:
        return jax.random.uniform(
            action_key, (env.generator.num_searchers, 2), minval=-1.0, maxval=1.0
        )

    check_env_does_not_smoke(env, select_action=select_action)


def test_env_specs_do_not_smoke(env: SearchAndRescue) -> None:
    """Test that we can access specs without any errors."""
    check_env_specs_does_not_smoke(env)


def test_target_detection(env: SearchAndRescue, key: chex.PRNGKey) -> None:
    # Keep targets in one location
    env._target_dynamics = RandomWalk(step_size=0.0)

    # Agent facing wrong direction should not see target
    state = State(
        searchers=AgentState(
            pos=jnp.array([[0.5, 0.5]]), heading=jnp.array([jnp.pi]), speed=jnp.array([0.0])
        ),
        targets=TargetState(pos=jnp.array([[0.54, 0.5]]), found=jnp.array([False])),
        key=key,
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

    # Searcher should only get rewards once
    state, timestep = env.step(state, jnp.zeros((1, 2)))
    assert state.targets.found[0]
    assert timestep.reward[0] == 0

    # Once detected target should remain detected if agent moves away
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


def test_multi_target_detection(env: SearchAndRescue, key: chex.PRNGKey) -> None:
    # Keep targets in one location
    env._target_dynamics = RandomWalk(step_size=0.0)
    env.searcher_params = AgentParams(
        max_rotate=0.1,
        max_accelerate=0.01,
        min_speed=0.01,
        max_speed=0.05,
        view_angle=0.25,
    )

    # Agent facing wrong direction should not see target
    state = State(
        searchers=AgentState(
            pos=jnp.array([[0.5, 0.5]]), heading=jnp.array([0.5 * jnp.pi]), speed=jnp.array([0.0])
        ),
        targets=TargetState(
            pos=jnp.array([[0.54, 0.5], [0.46, 0.5]]), found=jnp.array([False, False])
        ),
        key=key,
    )
    state, timestep = env.step(state, jnp.zeros((1, 2)))
    assert not state.targets.found[0]
    assert not state.targets.found[1]
    assert timestep.reward[0] == 0

    # Rotated agent should detect first target
    state = State(
        searchers=AgentState(
            pos=state.searchers.pos, heading=jnp.array([0.0]), speed=state.searchers.speed
        ),
        targets=state.targets,
        key=state.key,
    )
    state, timestep = env.step(state, jnp.zeros((1, 2)))
    assert state.targets.found[0]
    assert not state.targets.found[1]
    assert timestep.reward[0] == 1

    # Rotated agent should not detect another agent
    state = State(
        searchers=AgentState(
            pos=state.searchers.pos, heading=jnp.array([1.5 * jnp.pi]), speed=state.searchers.speed
        ),
        targets=state.targets,
        key=state.key,
    )
    state, timestep = env.step(state, jnp.zeros((1, 2)))
    assert state.targets.found[0]
    assert not state.targets.found[1]
    assert timestep.reward[0] == 0

    # Rotated agent again should see second agent
    state = State(
        searchers=AgentState(
            pos=state.searchers.pos, heading=jnp.array([jnp.pi]), speed=state.searchers.speed
        ),
        targets=state.targets,
        key=state.key,
    )
    state, timestep = env.step(state, jnp.zeros((1, 2)))
    assert state.targets.found[0]
    assert state.targets.found[1]
    assert timestep.reward[0] == 1


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
