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
from jumanji.environments.swarms.predator_prey import PredatorPrey
from jumanji.environments.swarms.predator_prey.rewards import (
    DistanceRewards,
    SparseRewards,
)
from jumanji.environments.swarms.predator_prey.types import (
    Actions,
    Observation,
    Rewards,
    State,
)
from jumanji.testing.env_not_smoke import (
    check_env_does_not_smoke,
    check_env_specs_does_not_smoke,
)
from jumanji.types import StepType, TimeStep

PREDATOR_VISION_RANGE = 0.2
PREY_VISION_RANGE = 0.1
PREDATOR_REWARD = 0.2
PREY_PENALTY = 0.1
AGENT_RADIUS = 0.05


@pytest.fixture
def env() -> PredatorPrey:
    return PredatorPrey(
        num_predators=2,
        num_prey=10,
        prey_vision_range=PREY_VISION_RANGE,
        predator_vision_range=PREDATOR_VISION_RANGE,
        num_vision=11,
        agent_radius=AGENT_RADIUS,
        sparse_rewards=True,
        predator_max_rotate=0.1,
        predator_max_accelerate=0.01,
        predator_min_speed=0.01,
        predator_max_speed=0.05,
        predator_view_angle=0.5,
        prey_max_rotate=0.1,
        prey_max_accelerate=0.01,
        prey_min_speed=0.01,
        prey_max_speed=0.05,
        prey_view_angle=0.5,
    )


def test_env_init(env: PredatorPrey) -> None:
    """
    Check newly initialised state has expected array shapes
    and initial timestep.
    """
    k = jax.random.PRNGKey(101)
    state, timestep = env.reset(k)
    assert isinstance(state, State)

    assert isinstance(state.predators, AgentState)
    assert state.predators.pos.shape == (env.num_predators, 2)
    assert state.predators.speed.shape == (env.num_predators,)
    assert state.predators.speed.shape == (env.num_predators,)

    assert isinstance(state.prey, AgentState)
    assert state.prey.pos.shape == (env.num_prey, 2)
    assert state.prey.speed.shape == (env.num_prey,)
    assert state.prey.speed.shape == (env.num_prey,)

    assert isinstance(timestep.observation, Observation)
    assert timestep.observation.predators.shape == (
        env.num_predators,
        2 * env.num_vision,
    )
    assert timestep.observation.prey.shape == (env.num_prey, 2 * env.num_vision)
    assert timestep.step_type == StepType.FIRST


@pytest.mark.parametrize("sparse_rewards", [True, False])
def test_env_step(env: PredatorPrey, sparse_rewards: bool) -> None:
    """
    Run several steps of the environment with random actions and
    check states (i.e. positions, heading, speeds) all fall
    inside expected ranges.
    """
    env.sparse_rewards = sparse_rewards
    key = jax.random.PRNGKey(101)
    n_steps = 22

    def step(
        carry: Tuple[chex.PRNGKey, State], _: None
    ) -> Tuple[Tuple[chex.PRNGKey, State], Tuple[State, TimeStep[Observation]]]:
        k, state = carry
        k, k_pred, k_prey = jax.random.split(k, num=3)
        actions = Actions(
            predators=jax.random.uniform(k_pred, (env.num_predators, 2), minval=-1.0, maxval=1.0),
            prey=jax.random.uniform(k_prey, (env.num_prey, 2), minval=-1.0, maxval=1.0),
        )
        new_state, timestep = env.step(state, actions)
        return (k, new_state), (state, timestep)

    init_state, _ = env.reset(key)
    (_, final_state), (state_history, timesteps) = jax.lax.scan(
        step, (key, init_state), length=n_steps
    )

    assert isinstance(state_history, State)

    assert state_history.predators.pos.shape == (n_steps, env.num_predators, 2)
    assert jnp.all((0.0 <= state_history.predators.pos) & (state_history.predators.pos <= 1.0))
    assert state_history.predators.speed.shape == (n_steps, env.num_predators)
    assert jnp.all(
        (env.predator_params.min_speed <= state_history.predators.speed)
        & (state_history.predators.speed <= env.predator_params.max_speed)
    )
    assert state_history.predators.speed.shape == (n_steps, env.num_predators)
    assert jnp.all(
        (0.0 <= state_history.predators.heading) & (state_history.predators.heading <= 2.0 * jnp.pi)
    )

    assert state_history.prey.pos.shape == (n_steps, env.num_prey, 2)
    assert jnp.all((0.0 <= state_history.prey.pos) & (state_history.prey.pos <= 1.0))
    assert state_history.prey.speed.shape == (n_steps, env.num_prey)
    assert jnp.all(
        (env.predator_params.min_speed <= state_history.prey.speed)
        & (state_history.prey.speed <= env.predator_params.max_speed)
    )
    assert state_history.prey.heading.shape == (n_steps, env.num_prey)
    assert jnp.all(
        (0.0 <= state_history.prey.heading) & (state_history.prey.heading <= 2.0 * jnp.pi)
    )


@pytest.mark.parametrize("sparse_rewards", [True, False])
def test_env_does_not_smoke(env: PredatorPrey, sparse_rewards: bool) -> None:
    """Test that we can run an episode without any errors."""
    env.sparse_rewards = sparse_rewards
    env.max_steps = 10

    def select_action(action_key: chex.PRNGKey, _state: Observation) -> Actions:
        predator_key, prey_key = jax.random.split(action_key)
        return Actions(
            predators=jax.random.uniform(
                predator_key, (env.num_predators, 2), minval=-1.0, maxval=1.0
            ),
            prey=jax.random.uniform(prey_key, (env.num_prey, 2), minval=-1.0, maxval=1.0),
        )

    check_env_does_not_smoke(env, select_action=select_action)


def test_env_specs_do_not_smoke(env: PredatorPrey) -> None:
    """Test that we can access specs without any errors."""
    check_env_specs_does_not_smoke(env)


@pytest.mark.parametrize(
    "predator_pos, predator_heading, predator_view, prey_pos, prey_heading, prey_view",
    [
        # Both out of view range
        ([[0.8, 0.5]], [jnp.pi], [(0, 0, 1.0)], [[0.2, 0.5]], [0.0], [(0, 0, 1.0)]),
        # In predator range but not prey
        ([[0.35, 0.5]], [jnp.pi], [(0, 5, 0.75)], [[0.2, 0.5]], [0.0], [(0, 0, 1.0)]),
        # Both view each other
        ([[0.25, 0.5]], [jnp.pi], [(0, 5, 0.25)], [[0.2, 0.5]], [0.0], [(0, 5, 0.5)]),
        # Prey facing wrong direction
        (
            [[0.25, 0.5]],
            [jnp.pi],
            [(0, 5, 0.25)],
            [[0.2, 0.5]],
            [jnp.pi],
            [(0, 0, 1.0)],
        ),
        # Prey sees closest predator
        (
            [[0.35, 0.5], [0.25, 0.5]],
            [jnp.pi, jnp.pi],
            [(0, 5, 0.75), (0, 16, 0.5), (1, 5, 0.25)],
            [[0.2, 0.5]],
            [0.0],
            [(0, 5, 0.5)],
        ),
        # Observed around wrapped edge
        (
            [[0.025, 0.5]],
            [jnp.pi],
            [(0, 5, 0.25)],
            [[0.975, 0.5]],
            [0.0],
            [(0, 5, 0.5)],
        ),
    ],
)
def test_view_observations(
    env: PredatorPrey,
    predator_pos: List[List[float]],
    predator_heading: List[float],
    predator_view: List[Tuple[int, int, float]],
    prey_pos: List[List[float]],
    prey_heading: List[float],
    prey_view: List[Tuple[int, int, float]],
) -> None:
    """
    Test view model generates expected array with different
    configurations of agents.
    """

    predator_pos = jnp.array(predator_pos)
    predator_heading = jnp.array(predator_heading)
    predator_speed = jnp.zeros(predator_heading.shape)

    prey_pos = jnp.array(prey_pos)
    prey_heading = jnp.array(prey_heading)
    prey_speed = jnp.zeros(prey_heading.shape)

    state = State(
        predators=AgentState(pos=predator_pos, heading=predator_heading, speed=predator_speed),
        prey=AgentState(pos=prey_pos, heading=prey_heading, speed=prey_speed),
        key=jax.random.PRNGKey(101),
    )

    obs = env._state_to_observation(state)

    assert isinstance(obs, Observation)

    predator_expected = jnp.ones(
        (
            predator_heading.shape[0],
            2 * env.num_vision,
        )
    )
    for i, idx, val in predator_view:
        predator_expected = predator_expected.at[i, idx].set(val)

    assert jnp.all(jnp.isclose(obs.predators, predator_expected))

    prey_expected = jnp.ones(
        (
            prey_heading.shape[0],
            2 * env.num_vision,
        )
    )
    for i, idx, val in prey_view:
        prey_expected = prey_expected.at[i, idx].set(val)

    assert jnp.all(jnp.isclose(obs.prey[0], prey_expected))


@pytest.mark.parametrize(
    "predator_pos, predator_reward, prey_pos, prey_reward",
    [
        ([0.5, 0.5], 0.0, [0.8, 0.5], 0.0),
        ([0.5, 0.5], PREDATOR_REWARD, [0.5999, 0.5], -PREY_PENALTY),
        ([0.5, 0.5], PREDATOR_REWARD, [0.5001, 0.5], -PREY_PENALTY),
    ],
)
def test_sparse_rewards(
    predator_pos: List[float],
    predator_reward: float,
    prey_pos: List[float],
    prey_reward: float,
) -> None:
    """
    Test sparse rewards are correctly assigned.
    """

    state = State(
        predators=AgentState(
            pos=jnp.array([predator_pos]),
            heading=jnp.zeros((1,)),
            speed=jnp.zeros((1,)),
        ),
        prey=AgentState(
            pos=jnp.array([prey_pos]),
            heading=jnp.zeros((1,)),
            speed=jnp.zeros((1,)),
        ),
        key=jax.random.PRNGKey(101),
    )

    reward_fn = SparseRewards(AGENT_RADIUS, PREDATOR_REWARD, PREY_PENALTY)
    rewards = reward_fn(state)

    assert isinstance(rewards, Rewards)
    assert rewards.predators[0] == predator_reward
    assert rewards.prey[0] == prey_reward


@pytest.mark.parametrize(
    "predator_pos, predator_reward, prey_pos, prey_reward",
    [
        ([0.5, 0.5], 0.0, [0.8, 0.5], 0.0),
        ([0.5, 0.5], 0.5 * PREDATOR_REWARD, [0.55, 0.5], -0.5 * PREY_PENALTY),
        ([0.5, 0.5], PREDATOR_REWARD, [0.5 + 1e-10, 0.5], -PREY_PENALTY),
    ],
)
def test_distance_rewards(
    predator_pos: List[float],
    predator_reward: float,
    prey_pos: List[float],
    prey_reward: float,
) -> None:
    """
    Test rewards scaled with distance are correctly assigned.
    """

    state = State(
        predators=AgentState(
            pos=jnp.array([predator_pos]),
            heading=jnp.zeros((1,)),
            speed=jnp.zeros((1,)),
        ),
        prey=AgentState(
            pos=jnp.array([prey_pos]),
            heading=jnp.zeros((1,)),
            speed=jnp.zeros((1,)),
        ),
        key=jax.random.PRNGKey(101),
    )

    reward_fn = DistanceRewards(
        PREDATOR_VISION_RANGE, PREY_VISION_RANGE, PREDATOR_REWARD, PREY_PENALTY
    )
    rewards = reward_fn(state)
    assert isinstance(rewards, Rewards)
    assert jnp.isclose(rewards.predators[0], predator_reward)
    assert jnp.isclose(rewards.prey[0], prey_reward)


def test_predator_prey_render(monkeypatch: pytest.MonkeyPatch, env: PredatorPrey) -> None:
    """Check that the render method builds the figure but does not display it."""
    monkeypatch.setattr(plt, "show", lambda fig: None)
    step_fn = jax.jit(env.step)
    state, timestep = env.reset(jax.random.PRNGKey(0))
    action = env.action_spec.generate_value()
    state, timestep = step_fn(state, action)
    env.render(state)
    env.close()


def test_snake__animation(env: PredatorPrey, tmpdir: py.path.local) -> None:
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
