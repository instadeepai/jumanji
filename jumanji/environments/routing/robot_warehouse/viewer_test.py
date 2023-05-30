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
import jax.random as random
import matplotlib
import matplotlib.pyplot as plt
import numpy as jnp
import py
import pytest

from jumanji.environments.routing.robot_warehouse import RobotWarehouse
from jumanji.environments.routing.robot_warehouse.viewer import RobotWarehouseViewer


def test_robot_warehouse_viewer__render(
    robot_warehouse_env: RobotWarehouse, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plt, "show", lambda fig: None)
    key = random.PRNGKey(0)
    state, _ = robot_warehouse_env.reset(key)
    grid_size = robot_warehouse_env._generator.grid_size
    goals = robot_warehouse_env._generator.goals

    viewer = RobotWarehouseViewer(grid_size, goals)
    viewer.render(state)
    viewer.close()


def test_robot_warehouse_viewer__animate(robot_warehouse_env: RobotWarehouse) -> None:
    key = random.PRNGKey(0)
    state, _ = jax.jit(robot_warehouse_env.reset)(key)
    grid_size = robot_warehouse_env._generator.grid_size
    goals = robot_warehouse_env._generator.goals

    num_steps = 5
    states = [state]
    for _ in range(num_steps - 1):
        key, subkey = jax.random.split(key)
        action = jax.random.choice(subkey, jnp.arange(5), shape=(2,))
        state, _ = jax.jit(robot_warehouse_env.step)(state, action)
        states.append(state)

    viewer = RobotWarehouseViewer(grid_size, goals)
    viewer.animate(states)
    viewer.close()


def test_robot_warehouse_viewer__save_animation(
    robot_warehouse_env: RobotWarehouse, tmpdir: py.path.local
) -> None:
    key = random.PRNGKey(0)
    state, _ = jax.jit(robot_warehouse_env.reset)(key)
    grid_size = robot_warehouse_env._generator.grid_size
    goals = robot_warehouse_env._generator.goals

    num_steps = 5
    states = [state]
    for _ in range(num_steps - 1):
        key, subkey = jax.random.split(key)
        action = jax.random.choice(subkey, jnp.arange(5), shape=(2,))
        state, _ = jax.jit(robot_warehouse_env.step)(state, action)
        states.append(state)

    viewer = RobotWarehouseViewer(grid_size, goals)
    animation = viewer.animate(states)
    assert isinstance(animation, matplotlib.animation.Animation)

    save_path = str(tmpdir.join("/robot_warehouse_animation_test.gif"))
    animation.save(save_path)
    viewer.close()
