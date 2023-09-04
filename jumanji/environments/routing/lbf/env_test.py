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
import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from jumanji.environments.routing.lbf.constants import DOWN, LOAD, NOOP, RIGHT
from jumanji.environments.routing.lbf.env import LevelBasedForaging
from jumanji.environments.routing.lbf.types import Agent, Food, State
from jumanji.testing.env_not_smoke import check_env_does_not_smoke
from jumanji.types import StepType


def test_get_reward(
    level_based_foraging_env: LevelBasedForaging, agents: Agent, foods: Food
) -> None:
    adj_food0_level = jnp.array([0.0, agents.level[1], agents.level[2], 0.0])
    adj_food1_level = jnp.array([0.0, 0.0, agents.level[2], 0.0])
    adj_agent_levels = jnp.array([adj_food0_level, adj_food1_level])
    eaten = jnp.array([True, False])

    reward = level_based_foraging_env.get_reward(foods, adj_agent_levels, eaten)

    expected_reward = (adj_food0_level * foods.level[0]) / (
        jnp.sum(foods.level) * jnp.sum(adj_food0_level)
    )

    assert jnp.all(reward == expected_reward)


def test__reward_per_food(
    level_based_foraging_env: LevelBasedForaging,
    agents: Agent,
    food0: Food,
    food1: Food,
) -> None:
    tot_food_level = food0.level + food1.level
    # what is the level of agents adjacent to food0
    adj_food0_level = jnp.array(
        [
            0.0,  # not adj
            agents.level[1],  # adj
            agents.level[2],  # adj
            0.0,  # not adj
        ]
    )

    # what is the level of agents adjacent to food1
    adj_food1_level = jnp.array(
        [
            0.0,  # not adj
            0.0,  # not adj
            agents.level[2],  # adj
            0.0,  # not adj
        ]
    )

    # check that reward is 0 if food is not eaten
    # food 0
    reward_not_eaten = level_based_foraging_env._reward_per_food(
        food0, adj_food0_level, jnp.asarray(False), tot_food_level
    )
    assert jnp.all(reward_not_eaten == 0.0)
    # food 1
    reward_not_eaten = level_based_foraging_env._reward_per_food(
        food1, adj_food1_level, jnp.asarray(False), tot_food_level
    )
    assert jnp.all(reward_not_eaten == 0.0)

    # check that correct reward received for food0 when eaten
    reward_eaten = level_based_foraging_env._reward_per_food(
        food0, adj_food0_level, jnp.asarray(True), tot_food_level
    )
    assert jnp.all(
        reward_eaten
        == (adj_food0_level * food0.level) / (jnp.sum(adj_food0_level) * tot_food_level)
    )

    # check that correct reward received for food1 when eaten
    reward_eaten = level_based_foraging_env._reward_per_food(
        food1, adj_food1_level, jnp.asarray(True), tot_food_level
    )
    assert jnp.all(
        reward_eaten
        == (adj_food1_level * food1.level) / (jnp.sum(adj_food1_level) * tot_food_level)
    )


def test_reset(level_based_foraging_env: LevelBasedForaging, key: chex.PRNGKey) -> None:
    num_agents = level_based_foraging_env._generator.num_agents
    grid_size = level_based_foraging_env._generator.grid_size

    state, timestep = level_based_foraging_env.reset(key)
    assert len(state.agents.position) == num_agents
    assert len(state.foods.position) == level_based_foraging_env._generator.num_food

    expected_obs_shape = (num_agents, 3, grid_size, grid_size)
    assert timestep.observation.agents_view.shape == expected_obs_shape

    assert jnp.all(timestep.discount == 1.0)
    assert jnp.all(timestep.reward == 0.0)
    assert timestep.step_type == StepType.FIRST

    assert timestep.discount.shape == (num_agents,)
    assert timestep.reward.shape == (num_agents,)


def test_step(level_based_foraging_env: LevelBasedForaging, state: State) -> None:
    # agent grid
    # [a0, a1, 0],
    # [a2, 0, a3],
    # [0, 0, 0],

    # food grid
    # [0, 0, 0],
    # [0, f0, 0],
    # [f1, 0, 0],

    num_agents = level_based_foraging_env._generator.num_agents
    foods = state.foods

    # tranisition where everyone does a no-op
    action = jnp.array([NOOP] * num_agents)
    next_state, timestep = level_based_foraging_env.step(state, action)

    assert jnp.all(timestep.discount == 1.0)
    assert jnp.all(timestep.reward == 0.0)
    assert timestep.discount.shape == (num_agents,)
    assert timestep.reward.shape == (num_agents,)
    assert timestep.step_type == StepType.MID

    assert next_state.step_count == state.step_count + 1

    chex.assert_trees_all_equal(next_state.foods, state.foods)
    chex.assert_trees_all_equal(next_state.agents, state.agents)

    # transition where all agents load food
    # middle food was eaten
    action = jnp.array([LOAD] * num_agents)
    next_state, next_timestep = level_based_foraging_env.step(state, action)
    assert jnp.all(next_timestep.discount == 1.0)
    # check reward is correct
    adj_levels = next_state.agents.level[jnp.array([1, 2, 3])]
    total_adj_level = jnp.sum(adj_levels)
    reward = (foods.level[0] * adj_levels) / (total_adj_level * jnp.sum(foods.level))
    reward = jnp.concatenate([jnp.array([0.0]), reward])  # add reward for agent 0

    assert jnp.all(next_timestep.reward == reward)
    assert next_timestep.discount.shape == (num_agents,)
    assert next_timestep.reward.shape == (num_agents,)

    # seeing as we loaded food and no one moved agent slice should look the same
    assert jnp.all(state.agents.position == next_state.agents.position)
    assert jnp.all(
        next_timestep.observation.agents_view[:, 0, ...]
        == timestep.observation.agents_view[:, 0, ...]
    )

    # check food positions
    expected_foods_view = jnp.array(
        [
            [[-1, -1, -1], [-1, 0, 0], [-1, 0, 0]],  # agent 0's food view
            [[-1, -1, -1], [0, 0, 0], [0, 0, 0]],  # agent 1's food view
            [[-1, 0, 0], [-1, 0, 0], [-1, 3, 0]],  # agent 2's food view
            [[0, 0, -1], [0, 0, -1], [0, 0, -1]],  # agent 3's food view
        ]
    )
    expected_mask_view = jnp.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 1]],  # agent 0's mask view
            [[0, 0, 0], [0, 1, 1], [0, 1, 0]],  # agent 1's mask view
            [[0, 0, 0], [0, 1, 1], [0, 0, 1]],  # agent 2's mask view
            [[0, 1, 0], [1, 1, 0], [1, 1, 0]],  # agent 3's mask view
        ]
    )
    assert jnp.all(next_state.foods.eaten == jnp.array([True, False]))
    assert jnp.all(
        next_timestep.observation.agents_view[:, 1, ...] == expected_foods_view
    )
    assert jnp.all(
        next_timestep.observation.agents_view[:, 2, ...] == expected_mask_view
    )

    # Test agents moving
    # Only agents 1, 2 and 3 have space to move
    action = jnp.array([NOOP, RIGHT, RIGHT, DOWN])
    next_state_1, next_timestep_1 = level_based_foraging_env.step(next_state, action)
    assert jnp.all(next_timestep_1.discount == 1.0)
    assert jnp.all(next_timestep_1.reward == 0.0)
    assert next_timestep_1.discount.shape == (num_agents,)
    assert next_timestep_1.reward.shape == (num_agents,)

    # check agent positions after move
    expected_agent_positions = jnp.array([[0, 0], [0, 2], [1, 1], [2, 2]])
    assert jnp.all(next_state_1.agents.position == expected_agent_positions)
    # todo: check agent positions in agent view


def test_step_done_horizon(
    level_based_foraging_env: LevelBasedForaging, key: chex.PRNGKey
) -> None:
    num_agents = level_based_foraging_env._generator.num_agents
    # test done after 5 steps
    state, timestep = level_based_foraging_env.reset(key)
    assert timestep.step_type == StepType.FIRST
    assert state.step_count == 0
    assert jnp.all(timestep.discount == 1.0)
    assert timestep.discount.shape == (num_agents,)
    assert timestep.reward.shape == (num_agents,)

    action = jnp.array([NOOP] * num_agents)
    state, timestep = level_based_foraging_env.step(state, action)

    for i in range(1, 5):
        assert timestep.step_type == StepType.MID
        assert state.step_count == i
        assert jnp.all(timestep.discount == 1.0)
        assert timestep.discount.shape == (num_agents,)
        assert timestep.reward.shape == (num_agents,)

        state, timestep = level_based_foraging_env.step(state, action)

    assert timestep.step_type == StepType.LAST
    assert state.step_count == 5
    assert jnp.all(timestep.discount == 0.0)
    assert timestep.discount.shape == (num_agents,)
    assert timestep.reward.shape == (num_agents,)


def test_step_done_all_eaten(
    level_based_foraging_env: LevelBasedForaging,
    agents: Agent,
    foods: Food,
    key: chex.PRNGKey,
) -> None:
    num_agents = level_based_foraging_env._generator.num_agents
    num_foods = level_based_foraging_env._generator.num_food

    # set agent 2's level high enough to eat food 1
    agents.level = agents.level.at[2].set(5)

    state = State(step_count=0, agents=agents, foods=foods, key=key)
    action = jnp.array([LOAD] * num_agents)
    state, timestep = level_based_foraging_env.step(state, action)

    assert timestep.step_type == StepType.LAST
    assert jnp.all(timestep.discount == 0.0)
    assert timestep.discount.shape == (num_agents,)
    assert timestep.reward.shape == (num_agents,)

    # check food positions
    assert jnp.all(state.foods.eaten)
    expected_foods_view = jnp.array(
        [
            [[-1, -1, -1], [-1, 0, 0], [-1, 0, 0]],  # agent 0's food view
            [[-1, -1, -1], [0, 0, 0], [0, 0, 0]],  # agent 1's food view
            [[-1, 0, 0], [-1, 0, 0], [-1, 0, 0]],  # agent 2's food view
            [[0, 0, -1], [0, 0, -1], [0, 0, -1]],  # agent 3's food view
        ]
    )
    expected_mask_view = jnp.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 1]],  # agent 0's mask view
            [[0, 0, 0], [0, 1, 1], [0, 1, 0]],  # agent 1's mask view
            [[0, 0, 0], [0, 1, 1], [0, 1, 1]],  # agent 2's mask view
            [[0, 1, 0], [1, 1, 0], [1, 1, 0]],  # agent 3's mask view
        ]
    )

    assert jnp.all(timestep.observation.agents_view[:, 1, ...] == expected_foods_view)
    assert jnp.all(timestep.observation.agents_view[:, 2, ...] == expected_mask_view)

    adj_levels_food_0 = state.agents.level[jnp.array([1, 2, 3])]
    total_adj_level_food_0 = jnp.sum(adj_levels_food_0)
    reward_food0 = (foods.level[0] * adj_levels_food_0) / (
        total_adj_level_food_0 * num_foods
    )
    # Add reward for agent 0.
    reward_food0 = jnp.concatenate([jnp.array([0.0]), reward_food0])

    adj_levels_food_1 = state.agents.level[2]
    total_adj_adj_level_food_1 = jnp.sum(adj_levels_food_1)
    reward_food1 = (foods.level[1] * adj_levels_food_1) / (
        total_adj_adj_level_food_1 * num_foods
    )
    # Add reward for agents 0, 1 and 3.
    reward_food1 = jnp.concatenate(
        [jnp.array([0.0, 0.0]), jnp.array([reward_food1]), jnp.array([0.0])]
    )

    # If all foods are eaten total reward is 1.
    assert jnp.sum(timestep.reward) == 1


def test_env_does_not_smoke(level_based_foraging_env: LevelBasedForaging) -> None:
    check_env_does_not_smoke(level_based_foraging_env)
