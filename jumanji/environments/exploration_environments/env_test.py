from typing import Type

import pytest

from jumanji.environments.exploration_environments.ant_maze import AntMaze
from jumanji.environments.exploration_environments.ant_trap import AntTrap
from jumanji.environments.exploration_environments.humanoid_trap import HumanoidTrap
from jumanji.environments.exploration_environments.point_maze import PointMaze


@pytest.mark.parametrize("env_cls", [AntMaze, AntTrap, HumanoidTrap, PointMaze])
def test_env(env_cls: Type) -> None:
    """Run random rollout on the environment."""
    horizon = 200
    env = env_cls()
    env.reset()
    done = False
    steps = 0
    while not done and steps < horizon:
        _, _, done, _ = env.step(env.action_space.sample())
        steps += 1
