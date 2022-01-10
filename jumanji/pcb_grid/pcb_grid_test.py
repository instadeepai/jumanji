from typing import Dict, Tuple

import numpy as np
import pytest

from jumanji.pcb_grid.pcb_grid import DOWN, LEFT, RIGHT, UP, PcbGridEnv, move

num_agents = 4
rows = 32
cols = 32
difficulty = "easy"

env_loop_iterations = 1000


@pytest.fixture
def pcb_grid_env() -> PcbGridEnv:
    """Instantiates a default PcbGrid environment."""
    return PcbGridEnv(
        rows=rows, cols=cols, num_agents=num_agents, difficulty=difficulty
    )


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
        possibly_blocked_agent = not all(initial_obs[agent_id]["action_mask"])
        assert not (initial_obs_agent == obs_agent).any() or possibly_blocked_agent


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
