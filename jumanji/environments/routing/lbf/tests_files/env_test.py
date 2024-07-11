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
import jax.numpy as jnp

from jumanji.environments.routing.lbf.constants import DOWN, LOAD, NOOP
from jumanji.environments.routing.lbf.env import LevelBasedForaging
from jumanji.environments.routing.lbf.types import Agent, Food, State
from jumanji.testing.env_not_smoke import check_env_does_not_smoke
from jumanji.types import StepType, TimeStep

# General integration test


def test_lbf_environment_integration(
    lbf_environment: LevelBasedForaging, key: chex.PRNGKey
) -> None:
    # Test the interaction of environment, agent, and food
    initial_state, timestep = lbf_environment.reset(key=key)
    assert isinstance(initial_state, State)
    assert isinstance(timestep, TimeStep)
    assert timestep.step_type == StepType.FIRST
    assert jnp.isclose(
        timestep.reward, jnp.zeros(lbf_environment.num_agents, dtype=float)
    ).all()
    assert timestep.extras == {"percent_eaten": jnp.float32(0)}
    # Test the step function
    action = jnp.array([NOOP] * lbf_environment.num_agents)
    next_state, timestep = lbf_environment.step(initial_state, action)
    assert isinstance(next_state, State)
    assert isinstance(timestep, TimeStep)
    assert timestep.step_type == StepType.MID


def test_reset(lbf_environment: LevelBasedForaging, key: chex.PRNGKey) -> None:
    num_agents = lbf_environment.num_agents
    num_food = lbf_environment.num_food

    state, timestep = lbf_environment.reset(key)
    assert len(state.agents.position) == num_agents
    assert len(state.food_items.position) == lbf_environment.num_food

    expected_obs_shape = (num_agents, 3 * (num_food + num_agents))
    assert timestep.observation.agents_view.shape == expected_obs_shape

    assert jnp.all(timestep.discount == 1.0)
    assert jnp.all(timestep.reward == 0.0)
    assert timestep.step_type == StepType.FIRST

    assert timestep.discount.shape == (num_agents,)
    assert timestep.reward.shape == (num_agents,)


def test_get_reward(
    lbf_environment: LevelBasedForaging, agents: Agent, food_items: Food
) -> None:
    adj_food0_level = jnp.array([0.0, agents.level[1], agents.level[2]])
    adj_food1_level = jnp.array([0.0, 0.0, agents.level[2]])
    adj_food2_level = jnp.array([0.0, 0.0, 0.0])
    adj_agent_levels = jnp.array([adj_food0_level, adj_food1_level, adj_food2_level])
    eaten = jnp.array([True, True, False])

    reward = lbf_environment.get_reward(food_items, adj_agent_levels, eaten)

    expected_reward_food0 = (adj_food0_level * food_items.level[0]) / (
        jnp.sum(food_items.level) * jnp.sum(adj_food0_level)
    )
    expected_reward_food1 = (adj_food1_level * food_items.level[1]) / (
        jnp.sum(food_items.level) * jnp.sum(adj_food1_level)
    )
    expected_reward = expected_reward_food0 + expected_reward_food1
    assert jnp.all(reward == expected_reward)


def test_step(lbf_environment: LevelBasedForaging, state: State) -> None:

    num_agents = lbf_environment.num_agents
    ep_return = jnp.zeros((num_agents,), jnp.int32)

    # tranisition where everyone does a no-op
    action = jnp.array([NOOP] * num_agents)
    next_state, timestep = lbf_environment.step(state, action)

    assert jnp.all(timestep.discount == 1.0)
    assert jnp.all(timestep.reward == 0.0)
    assert timestep.discount.shape == (num_agents,)
    assert timestep.reward.shape == (num_agents,)
    assert timestep.step_type == StepType.MID
    assert next_state.step_count == state.step_count + 1

    chex.assert_trees_all_equal(next_state.food_items, state.food_items)
    chex.assert_trees_all_equal(next_state.agents, state.agents)

    # transition where all agents load food: eat food0 and food1
    action = jnp.array([LOAD] * num_agents)
    next_state, next_timestep = lbf_environment.step(state, action)
    ep_return += next_timestep.reward
    assert jnp.all(next_state.food_items.eaten == jnp.array([True, True, False]))

    # End the episode by eating the last food item: Agent2 eats food2
    action = jnp.array([DOWN] * num_agents)
    next_state, next_timestep = lbf_environment.step(next_state, action)
    action = jnp.array([LOAD] * num_agents)
    next_state, next_timestep = lbf_environment.step(next_state, action)
    ep_return += next_timestep.reward

    assert jnp.all(next_state.food_items.eaten == jnp.array([True, True, True]))
    assert next_timestep.step_type == StepType.LAST
    assert jnp.all(next_timestep.discount == 0.0)
    # If all foods are eaten total reward is 1.
    assert jnp.sum(ep_return) == 1


def test_step_done_horizon(
    lbf_environment: LevelBasedForaging, key: chex.PRNGKey
) -> None:
    num_agents = lbf_environment.num_agents
    # Test the done after 5 steps
    state, timestep = lbf_environment.reset(key)
    assert timestep.step_type == StepType.FIRST
    assert state.step_count == 0
    assert jnp.all(timestep.discount == 1.0)
    assert timestep.discount.shape == (num_agents,)
    assert timestep.reward.shape == (num_agents,)

    action = jnp.array([NOOP] * num_agents)
    state, timestep = lbf_environment.step(state, action)

    for i in range(1, 5):
        assert timestep.step_type == StepType.MID
        assert state.step_count == i
        assert jnp.all(timestep.discount == 1.0)
        assert timestep.discount.shape == (num_agents,)
        assert timestep.reward.shape == (num_agents,)

        state, timestep = lbf_environment.step(state, action)

    assert timestep.step_type == StepType.LAST
    assert state.step_count == 5
    assert jnp.all(timestep.discount == 1.0)
    assert timestep.discount.shape == (num_agents,)
    assert timestep.reward.shape == (num_agents,)


def test_env_does_not_smoke(
    lbf_environment: LevelBasedForaging, lbf_env_grid_obs: LevelBasedForaging
) -> None:
    check_env_does_not_smoke(lbf_environment)
    check_env_does_not_smoke(lbf_env_grid_obs)
