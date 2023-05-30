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

from typing import Tuple

import jax
import jax.numpy as jnp
import pytest

from jumanji.environments.routing.robot_warehouse import RobotWarehouse
from jumanji.environments.routing.robot_warehouse.generator import RandomGenerator
from jumanji.environments.routing.robot_warehouse.types import (
    Agent,
    Position,
    Shelf,
    State,
)
from jumanji.types import TimeStep


@pytest.fixture(scope="module")
def robot_warehouse_env() -> RobotWarehouse:
    """Instantiates a default RobotWarehouse environment with 2 agents, 1 shelf row, 3 shelf columns,
    a column height of 2, sensor range of 1 and a request queue size of 4."""
    generator = RandomGenerator(
        shelf_rows=1,
        shelf_columns=3,
        column_height=2,
        num_agents=2,
        sensor_range=1,
        request_queue_size=4,
    )

    env = RobotWarehouse(
        generator=generator,
        time_limit=5,
    )
    return env


@pytest.fixture
def deterministic_robot_warehouse_env(
    robot_warehouse_env: RobotWarehouse,
) -> Tuple[RobotWarehouse, State, TimeStep]:
    """Instantiates a RobotWarehouse environment with 2 agents and 8 shelves
    with a step limit of 5."""
    state, timestep = robot_warehouse_env.reset(jax.random.PRNGKey(42))

    # create agents, shelves and grid
    def make_agent(x: int, y: int, direction: int, is_carrying: int) -> Agent:
        return Agent(Position(x=x, y=y), direction=direction, is_carrying=is_carrying)

    def make_shelf(x: int, y: int, is_requested: int) -> Shelf:
        return Shelf(Position(x=x, y=y), is_requested=is_requested)

    # agent information
    xs = jnp.array([3, 1])
    ys = jnp.array([4, 7])
    dirs = jnp.array([2, 3])
    carries = jnp.array([0, 0])
    state.agents = jax.vmap(make_agent)(xs, ys, dirs, carries)

    # shelf information
    xs = jnp.array([1, 1, 1, 1, 2, 2, 2, 2])
    ys = jnp.array([1, 2, 7, 8, 1, 2, 7, 8])
    requested = jnp.array([0, 1, 1, 0, 0, 0, 1, 1])
    state.shelves = jax.vmap(make_shelf)(xs, ys, requested)

    # create grid
    state.grid = jnp.array(
        [
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 2, 0, 0, 0, 0, 3, 4, 0],
                [0, 5, 6, 0, 0, 0, 0, 7, 8, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ]
    )
    return robot_warehouse_env, state, timestep
