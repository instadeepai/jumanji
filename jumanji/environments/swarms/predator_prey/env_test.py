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
import pytest

from jumanji.environments.swarms.common.types import AgentState
from jumanji.environments.swarms.predator_prey import PredatorPrey
from jumanji.environments.swarms.predator_prey.types import Actions, Observation, State
from jumanji.types import StepType, TimeStep


@pytest.fixture
def env() -> PredatorPrey:
    return PredatorPrey(
        num_predators=2,
        num_prey=10,
        prey_vision_range=0.1,
        predator_vision_range=0.1,
        num_vision=10,
        agent_radius=0.01,
        sparse_rewards=True,
        prey_penalty=0.1,
        predator_rewards=0.2,
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


def test_env_step(env: PredatorPrey) -> None:
    key = jax.random.PRNGKey(101)
    n_steps = 11

    def step(
        carry: Tuple[chex.PRNGKey, State], _: None
    ) -> Tuple[Tuple[chex.PRNGKey, State], Tuple[State, TimeStep[Observation]]]:
        k, state = carry
        k, k_pred, k_prey = jax.random.split(k, num=3)
        actions = Actions(
            predators=jax.random.uniform(
                k_pred, (env.num_predators, 2), minval=-1.0, maxval=1.0
            ),
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
    assert jnp.all(
        (0.0 <= state_history.predators.pos) & (state_history.predators.pos <= 1.0)
    )
    assert state_history.predators.speed.shape == (n_steps, env.num_predators)
    assert jnp.all(
        (env.predator_params.min_speed <= state_history.predators.speed)
        & (state_history.predators.speed <= env.predator_params.max_speed)
    )
    assert state_history.prey.speed.shape == (n_steps, env.num_predators)
    assert jnp.all(
        (0.0 <= state_history.predators.heading)
        & (state_history.predators.heading <= 2.0 * jnp.pi)
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
        (0.0 <= state_history.prey.heading)
        & (state_history.prey.heading <= 2.0 * jnp.pi)
    )
