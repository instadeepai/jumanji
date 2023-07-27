import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.lbf.constants import DOWN, LEFT, RIGHT
from jumanji.environments.routing.lbf.types import Agent, Food
from jumanji.environments.routing.lbf.utils import eat, fix_collisions, is_adj, move


def test_move(agent1: Agent, foods: Food) -> None:
    # agent 1 is at [0, 1] and can move to [0, 0] or [0, 2].
    # But there is a food at [1, 1] so it cannot move there.
    grid_size = 3

    # move to [0, 0]
    agent1_new = move(agent1, LEFT, foods, grid_size)
    assert jnp.all(agent1_new.position == jnp.array([0, 0]))

    # move agent twice from [0, 0] to [2, 0] (where food is)
    agent1_new = move(agent1_new, DOWN, foods, grid_size)
    agent1_new = move(agent1_new, DOWN, foods, grid_size)
    assert jnp.all(agent1_new.position == jnp.array([1, 0]))

    # move agent to [0, 2]
    agent1_new = move(agent1, RIGHT, foods, grid_size)
    assert jnp.all(agent1_new.position == jnp.array([0, 2]))

    # try move agent to [1, 1] (where food is)
    agent1_new = move(agent1, DOWN, foods, grid_size)
    assert jnp.all(agent1_new.position == jnp.array([0, 1]))


def test_is_adj(
    agents: Agent, agent0: Agent, agent1: Agent, agent2: Agent, agent3: Agent
):
    assert is_adj(agent0, agent1)
    assert is_adj(agent0, agent2)
    assert not is_adj(agent0, agent3)
    assert not is_adj(agent1, agent3)
    assert not is_adj(agent2, agent3)
    assert not is_adj(agent1, agent2)

    # check that vmap also works with is_adj
    expected_adj = jnp.array([False, True, True, False])
    assert jnp.all(jax.vmap(is_adj, (0, None))(agents, agent0) == expected_adj)


def test_eat(agents: Agent, food0: Food, food1: Food) -> None:
    # food 0 can be eaten, food 1 has too high a level for adj agents.
    # set all agent actions to loading
    all_loading_agents = jax.vmap(lambda agent: agent.replace(loading=True))(agents)

    # check that food 0 can be eaten
    new_food0, eaten_food0, adj_agents = eat(all_loading_agents, food0)
    assert new_food0.eaten == True
    assert eaten_food0 == True
    assert jnp.all(adj_agents == jnp.array([False, True, True, True]))

    # check that food 1 cannot be eaten
    new_food1, eaten_food1, adj_agents = eat(all_loading_agents, food1)
    assert new_food1.eaten == False
    assert eaten_food1 == False
    assert jnp.all(adj_agents == jnp.array([False, False, True, False]))

    # check that if food is already eaten, it cannot be eaten again
    new_food0, eaten_food0, adj_agents = eat(all_loading_agents, new_food0)
    assert new_food0.eaten == True
    assert eaten_food0 == False
    assert jnp.all(adj_agents == jnp.array([False, True, True, True]))


def test_flag_duplicates() -> None:
    pass


def test_fix_collisions(agents: Agent):
    # agents original postions: [[0, 0], [0, 1], [1, 0], [1, 2]]

    # fake moves for agents:
    moved_agents = jax.vmap(Agent)(
        id=agents.id,
        level=agents.level,
        # collision on agent 0 and 3
        position=jnp.array([[0, 0], [0, 2], [2, 0], [0, 2]]),
    )

    # expected postions after collision fix:
    expected_agents = jax.vmap(Agent)(
        id=agents.id,
        level=agents.level,
        # take orig agent for agent 0 and 3
        position=jnp.array([[0, 0], [0, 1], [2, 0], [1, 2]]),
    )

    new_agents = fix_collisions(moved_agents, agents)
    chex.assert_trees_all_equal(new_agents, expected_agents)
