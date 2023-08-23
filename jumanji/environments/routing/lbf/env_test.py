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

import jax
import jax.numpy as jnp

from jumanji.environments.routing.lbf.env import LevelBasedForaging
from jumanji.environments.routing.lbf.types import Agent, Food, State
from jumanji.testing.env_not_smoke import check_env_does_not_smoke


def test_get_reward(
    level_based_foraging_env: LevelBasedForaging, agents: Agent, foods: Food
) -> None:
    adj_food0_level = jnp.array([0.0, agents.level[1], agents.level[2], 0.0])
    adj_food1_level = jnp.array([0.0, 0.0, agents.level[2], 0.0])
    adj_agent_levels = jnp.array([adj_food0_level, adj_food1_level])
    eaten = jnp.array([True, False])

    reward = level_based_foraging_env.get_reward(foods, adj_agent_levels, eaten)

    expected_reward = (adj_food0_level * foods.level[0]) / (
        level_based_foraging_env._generator.num_food * jnp.sum(adj_food0_level)
    )

    assert jnp.all(reward == expected_reward)


def test__reward_per_food(
    level_based_foraging_env: LevelBasedForaging,
    agents: Agent,
    food0: Food,
    food1: Food,
) -> None:
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
    reward_not_eaten = level_based_foraging_env._reward_per_food(
        food0, adj_food0_level, jnp.asarray(False)
    )
    assert jnp.all(reward_not_eaten == 0.0)

    reward_not_eaten = level_based_foraging_env._reward_per_food(
        food1, adj_food1_level, jnp.asarray(False)
    )
    assert jnp.all(reward_not_eaten == 0.0)

    # check that correct reward received for food0
    reward_eaten = level_based_foraging_env._reward_per_food(
        food0, adj_food0_level, jnp.asarray(True)
    )
    assert jnp.all(
        reward_eaten
        == (adj_food0_level * food0.level)
        / (level_based_foraging_env._generator.num_food * jnp.sum(adj_food0_level))
    )

    # check that correct reward received for food1
    reward_eaten = level_based_foraging_env._reward_per_food(
        food1, adj_food1_level, jnp.asarray(True)
    )
    assert jnp.all(
        reward_eaten
        == (adj_food1_level * food1.level)
        / (level_based_foraging_env._generator.num_food * jnp.sum(adj_food1_level))
    )


def test__state_to_obs(
    level_based_foraging_env: LevelBasedForaging, agents: Agent, foods: Food
) -> None:
    # agent grid
    # [1, 2, 0],
    # [2, 0, 1],
    # [0, 0, 0],

    # food grid
    # [0, 0, 0],
    # [0, 4, 0],
    # [3, 0, 0],
    key = jax.random.PRNGKey(0)
    state = State(step_count=jnp.asarray(0), agents=agents, foods=foods, key=key)
    obs = level_based_foraging_env._state_to_obs(state)
    expected_agent_0_view = jnp.array(
        [
            [
                # other agent levels
                [-1, -1, -1],
                [-1, 1, 2],
                [-1, 2, 0],
            ],
            [
                # food levels
                [-1, -1, -1],
                [-1, 0, 0],
                [-1, 0, 4],
            ],
            [
                # access (where can the agent go?)
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
        ]
    )

    assert jnp.all(obs.agent_views[0, ...] == expected_agent_0_view)
    assert jnp.all(
        obs.action_mask[0, ...] == jnp.array([True, False, False, False, False, True])
    )

    expected_agent_1_view = jnp.array(
        [
            [
                [-1, -1, -1],
                [1, 2, 0],
                [2, 0, 1],
            ],
            [
                [-1, -1, -1],
                [0, 0, 0],
                [0, 4, 0],
            ],
            [
                [0, 0, 0],
                [0, 1, 1],
                [0, 0, 0],
            ],
        ]
    )
    assert jnp.all(obs.agent_views[1, ...] == expected_agent_1_view)
    assert jnp.all(
        obs.action_mask[1, ...] == jnp.array([True, False, True, False, False, True])
    )


def test_reset(level_based_foraging_env: LevelBasedForaging) -> None:
    pass


def test_step(level_based_foraging_env: LevelBasedForaging) -> None:
    pass


def test_step_done(level_based_foraging_env: LevelBasedForaging) -> None:
    pass


def test_env_does_not_smoke(level_based_foraging_env: LevelBasedForaging) -> None:
    check_env_does_not_smoke(level_based_foraging_env)
