from typing import Type

import pytest
from pyvirtualdisplay import Display

from jumanji.mujoco.ant_maze import AntMaze
from jumanji.mujoco.ant_trap import AntTrap
from jumanji.mujoco.humanoid_trap import HumanoidTrap
from jumanji.mujoco.point_maze import PointMaze


@pytest.fixture(scope="module")
def display() -> Display:
    display = Display(visible=0, size=(1400, 900))
    yield display.start()
    display.stop()


@pytest.mark.parametrize("env_cls", [AntMaze, AntTrap, HumanoidTrap, PointMaze])
def test_env(display: Display, env_cls: Type) -> None:
    """Run random rollout on the environment."""
    horizon = 200
    env = env_cls()
    env.reset()
    done = False
    steps = 0
    while not done and steps < horizon:
        _, _, done, _ = env.step(env.action_space.sample())
        env.render()
        steps += 1
    env.close()
