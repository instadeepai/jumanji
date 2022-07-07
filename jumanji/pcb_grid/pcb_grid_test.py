from typing import Dict, Tuple

import numpy as np
import pytest
from pyvirtualdisplay import Display

from jumanji.pcb_grid.pcb_grid import (
    DOWN,
    HEAD,
    LEFT,
    RIGHT,
    SOURCE,
    TARGET,
    UP,
    PcbGridEnv,
    move,
)

num_agents = 4
rows = 32
cols = 32
width = 1000
height = 1000
difficulty = "easy"

env_loop_iterations = 1000


@pytest.fixture
def pcb_grid_env() -> PcbGridEnv:
    """Instantiates a default PcbGrid environment."""
    return PcbGridEnv(
        rows=rows, cols=cols, num_agents=num_agents, difficulty=difficulty
    )


@pytest.fixture(scope="module")
def display() -> Display:
    """Creates a virtual display so that a GUI is not displayed during testing."""
    display = Display(visible=False, size=(width, height))
    yield display.start()
    display.stop()


@pytest.mark.parametrize("pcb_grid_env", [()], indirect=True)
def test_pcb_grid__reset(pcb_grid_env: PcbGridEnv) -> None:
    """Validates the reset of the environment."""
    agent_obs = pcb_grid_env.reset()

    # Check obs conform
    assert isinstance(agent_obs, Dict)
    assert isinstance(agent_obs[0], Dict)

    # Check env correct dimensions
    assert len(pcb_grid_env.agents) == num_agents
    assert pcb_grid_env.grid.shape == (rows, cols)

    # Check no overlapping start positions / targets
    start_positions = [agent.position for agent in pcb_grid_env.agents]
    assert len(start_positions) == len(set(start_positions))
    targets = [agent.target for agent in pcb_grid_env.agents]
    assert len(targets) == len(set(targets))


@pytest.mark.parametrize("pcb_grid_env", [()], indirect=True)
def test_pcb_grid__step(pcb_grid_env: PcbGridEnv) -> None:
    """Validates the step function of the environment."""
    initial_obs = pcb_grid_env.reset()

    actions = {agent_id: np.random.randint(1, 5) for agent_id in range(num_agents)}
    env_tuple: Tuple = pcb_grid_env.step(actions)
    obs, reward, done, extras = env_tuple

    # Check env tuple types
    assert isinstance(obs, Dict)
    assert isinstance(obs[0], Dict)
    assert isinstance(reward, Dict)
    assert isinstance(done, Dict)
    assert isinstance(extras, Dict)

    # Check that agent obs have changed, allow for agent being possibly blocked
    for agent_id in range(num_agents):
        initial_obs_agent = initial_obs[agent_id]["image"]
        obs_agent = obs[agent_id]["image"]
        is_agent_blocked = not all(initial_obs[agent_id]["action_mask"][1:])
        assert not (initial_obs_agent == obs_agent).all() or is_agent_blocked


@pytest.mark.parametrize("pcb_grid_env", [()], indirect=True)
def test_pcb_grid__render(pcb_grid_env: PcbGridEnv, display: Display) -> None:
    """Validates that the render function of PcbGridEnv doesn't cause an error."""
    pcb_grid_env.reset()
    pcb_grid_env.render()


@pytest.mark.parametrize("pcb_grid_env", [()], indirect=True)
def test_pcb_grid__close(pcb_grid_env: PcbGridEnv, display: Display) -> None:
    """Validates the close method of PcbGridEnv."""
    pcb_grid_env.reset()
    pcb_grid_env.render()
    pcb_grid_env.close()


@pytest.mark.parametrize("pcb_grid_env", [()], indirect=True)
def test_pcb_grid__does_not_smoke(
    pcb_grid_env: PcbGridEnv, capsys: pytest.CaptureFixture
) -> None:
    """Test that we can run the environment loop without any errors."""
    pcb_grid_env.reset()

    print("RUNNING")
    for _ in range(env_loop_iterations):
        actions = {agent_id: np.random.randint(1, 5) for agent_id in range(num_agents)}
        env_tuple: Tuple = pcb_grid_env.step(actions)
        obs, reward, done, _ = env_tuple
        if done["__all__"]:
            pcb_grid_env.reset()
    assert capsys.readouterr().out


def test_move() -> None:
    """Validates move method from utils.
    Checks that starting from a certain position, taking some actions
    lead to the correct new positions.
    """
    position = (0, 0)
    actions = [LEFT, RIGHT, UP, DOWN]
    updated_positions = [move(position, action) for action in actions]
    next_poss = [
        (0, -1),
        (0, 1),
        (-1, 0),
        (1, 0),
    ]
    assert updated_positions == next_poss


@pytest.mark.parametrize("pcb_grid_env", [()], indirect=True)
def test_pcb_grid__no_nan(pcb_grid_env: PcbGridEnv) -> None:
    """Validates that no nan is encountered in either the environment returns
    throughout an episode.
    """
    pcb_grid_env.reset()

    # Run env loop
    for _ in range(env_loop_iterations):
        actions = {agent_id: np.random.randint(1, 5) for agent_id in range(num_agents)}
        env_tuple: Tuple = pcb_grid_env.step(actions)
        obs, reward, done, extras = env_tuple
        if done["__all__"]:
            pcb_grid_env.reset()

        # Check env tuple types
        assert isinstance(obs, Dict)
        assert isinstance(obs[0], Dict)
        assert isinstance(reward, Dict)
        assert isinstance(done, Dict)
        assert isinstance(extras, Dict)


@pytest.mark.parametrize("pcb_grid_env", [()], indirect=True)
def test_pcb_grid__agent_observations(pcb_grid_env: PcbGridEnv) -> None:
    """Test that the agents always get the correct observations."""
    pcb_grid_env.reset()
    # do random actions to populate the board with wires
    for _ in range(10):
        actions = {agent_id: np.random.randint(1, 5) for agent_id in range(num_agents)}
        obs, *_ = pcb_grid_env.step(actions)

    agent_0_obs = obs[0]["image"]

    # for each agent, check that the position of its SOURCE, HEAD and TARGET are the same
    # from the perspective of agent 0 and from its own perspective
    cell_types = [SOURCE, HEAD, TARGET]
    for agent_id in range(num_agents):
        curr_agent_obs = obs[agent_id]["image"]

        for cell_type in cell_types:
            assert np.all(
                (agent_0_obs == cell_type + 3 * agent_id)
                == (curr_agent_obs == cell_type)
            )


@pytest.mark.parametrize("pcb_grid_env", [()], indirect=True)
def test_pcb_grid__rewards_stop(pcb_grid_env: PcbGridEnv) -> None:
    pcb_grid_env.reset()
    for _ in range(5):
        _, reward, *_ = pcb_grid_env.step(
            {i: 0 for i in range(pcb_grid_env.num_agents)}
        )
        assert reward == {
            i: pcb_grid_env.reward_per_timestep + pcb_grid_env.reward_per_noop
            for i in range(pcb_grid_env.num_agents)
        }

    assert not all(pcb_grid_env._previous_dones.values())

    # forcing all agents to finish
    for agent in pcb_grid_env.agents:
        agent.position = agent.target
        pcb_grid_env.grid[agent.target] = HEAD + 3 * agent.agent_id

    # all connected
    _, reward, *_ = pcb_grid_env.step({i: 0 for i in range(pcb_grid_env.num_agents)})
    assert reward == {
        i: pcb_grid_env.reward_per_connected for i in range(pcb_grid_env.num_agents)
    }
    assert all(pcb_grid_env._previous_dones.values())

    # already connected should get reward of 0
    _, reward, *_ = pcb_grid_env.step({i: 0 for i in range(pcb_grid_env.num_agents)})
    assert reward == {i: 0 for i in range(pcb_grid_env.num_agents)}
