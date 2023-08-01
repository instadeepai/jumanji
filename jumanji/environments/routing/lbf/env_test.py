import jax.numpy as jnp

from jumanji.environments.routing.lbf.env import LevelBasedForaging
from jumanji.environments.routing.lbf.types import Agent, Food


def test__reward_per_food(
    level_based_foraging_env: LevelBasedForaging,
    agents: Agent,
    food0: Food,
    food1: Food,
):
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


def test_get_reward(
    level_based_foraging_env: LevelBasedForaging, agents: Agent, foods: Food
):
    adj_food0_level = jnp.array([0.0, agents.level[1], agents.level[2], 0.0])
    adj_food1_level = jnp.array([0.0, 0.0, agents.level[2], 0.0])
    adj_agent_levels = jnp.array([adj_food0_level, adj_food1_level])
    eaten = jnp.array([True, False])

    reward = level_based_foraging_env.get_reward(foods, adj_agent_levels, eaten)

    expected_reward = (adj_food0_level * foods.level[0]) / (
        level_based_foraging_env._generator.num_food * jnp.sum(adj_food0_level)
    )

    assert jnp.all(reward == expected_reward)


def test__state_to_timestep(level_based_foraging_env: LevelBasedForaging):
    pass


def test__get_agent_view(level_based_foraging_env: LevelBasedForaging):
    pass


def test_reset(level_based_foraging_env: LevelBasedForaging):
    pass


def test_step(level_based_foraging_env: LevelBasedForaging):
    pass


def test_step_done(level_based_foraging_env: LevelBasedForaging):
    pass


def test_env_does_not_smoke(level_based_foraging_env: LevelBasedForaging):
    pass
