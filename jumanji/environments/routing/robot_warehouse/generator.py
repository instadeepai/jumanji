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

import abc
from typing import Callable

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.robot_warehouse.types import State
from jumanji.environments.routing.robot_warehouse.utils import compute_action_mask
from jumanji.environments.routing.robot_warehouse.utils_spawn import (
    place_entities_on_grid,
    spawn_random_entities,
)


class Generator(abc.ABC):
    """Base class for generators for the RobotWarehouse environment."""

    def __init__(
        self,
        shelf_rows: int,
        shelf_columns: int,
        column_height: int,
        num_agents: int,
        sensor_range: int,
        request_queue_size: int,
    ) -> None:
        """Initializes a robot_warehouse generator, used to generate grids for
        the RobotWarehouse environment.

        Args:
            shelf_rows: the number of shelf cluster rows, each of height = column_height.
                Defaults to 1.
            shelf_columns: the number of shelf cluster columns, each of width = 2 cells
                (must be an odd number). Defaults to 3.
            column_height: the height of each shelf cluster. Defaults to 8.
            num_agents: the number of agents (robots) operating on the warehouse floor.
                Defaults to 2.
            sensor_range: the receptive field around an agent          O O O
                (e.g. 1 implies a 360 view of 1 cell around the    ->  O x O
                agent's position cell)                                 O O O
                Defaults to 1.
            request_queue_size: the number of shelves requested at any
                given time which remains fixed throughout environment steps. Defaults to 4.
        """
        if shelf_columns % 2 != 1:
            raise ValueError(
                "Environment argument: `shelf_columns`, must be an odd number."
            )

        self._shelf_rows = shelf_rows
        self._shelf_columns = shelf_columns
        self._column_height = column_height
        self._num_agents = num_agents
        self._sensor_range = sensor_range
        self._request_queue_size = request_queue_size

        self._grid_size = (
            (column_height + 1) * shelf_rows + 2,
            (2 + 1) * shelf_columns + 1,
        )
        self._agent_ids = jnp.arange(num_agents)

    @property
    def shelf_rows(self) -> int:
        return self._shelf_rows

    @property
    def shelf_columns(self) -> int:
        return self._shelf_columns

    @property
    def column_height(self) -> int:
        return self._column_height

    @property
    def grid_size(self) -> chex.Array:
        return self._grid_size

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def sensor_range(self) -> int:
        return self._sensor_range

    @property
    def request_queue_size(self) -> int:
        return self._request_queue_size

    @property
    def agent_ids(self) -> chex.Array:
        return self._agent_ids

    @property
    @abc.abstractmethod
    def shelf_ids(self) -> chex.Array:
        """shelf ids"""

    @property
    @abc.abstractmethod
    def not_in_queue_size(self) -> chex.Array:
        """number of shelves not in queue"""

    @property
    @abc.abstractmethod
    def highways(self) -> chex.Array:
        """highways positions"""

    @property
    @abc.abstractmethod
    def goals(self) -> chex.Array:
        """goals positions"""

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates an `RobotWarehouse` state.

        Returns:
            An `RobotWarehouse` state.
        """


class GeneratorBase(Generator):
    """Base class for `RobotWarehouse` environment state generator."""

    def __init__(
        self,
        shelf_rows: int,
        shelf_columns: int,
        column_height: int,
        num_agents: int,
        sensor_range: int,
        request_queue_size: int,
    ) -> None:
        """Initializes a robot_warehouse generator."""
        super().__init__(
            shelf_rows,
            shelf_columns,
            column_height,
            num_agents,
            sensor_range,
            request_queue_size,
        )
        self._make_warehouse()

    def _make_warehouse(self) -> None:
        """Create the layout for the warehouse floor, i.e. the grid

        Args:
            shelf_rows: the number of shelf cluster rows
            shelf_columns: the number of shelf cluster columns
            column_height: the height of each shelf cluster
        """

        # create goal positions
        self._goals = jnp.array(
            [
                (self._grid_size[1] // 2 - 1, self._grid_size[0] - 1),
                (self._grid_size[1] // 2, self._grid_size[0] - 1),
            ]
        )
        # calculate "highways" (these are open spaces/cells between shelves)
        highway_func: Callable[[int, int], bool] = lambda x, y: (
            (y % 3 == 0)  # vertical highways
            | (x % (self.column_height + 1) == 0)  # horizontal highways
            | (x == self._grid_size[0] - 1)  # delivery row
            | (  # remove middle cluster to allow agents to queue in front of goals
                (x > self._grid_size[0] - (self.column_height + 3))
                & ((y == self._grid_size[1] // 2 - 1) | (y == self._grid_size[1] // 2))
            )
        )
        grid_indices = jnp.indices(jnp.zeros(self._grid_size, dtype=jnp.int32).shape)
        self._highways = jax.vmap(highway_func)(grid_indices[0], grid_indices[1])

        non_highways = jnp.abs(self.highways - 1)

        # shelves information
        n_shelves = jnp.sum(non_highways)
        self._shelf_positions = jnp.argwhere(non_highways)
        self._shelf_ids = jnp.arange(n_shelves)
        self._not_in_queue_size = n_shelves - self.request_queue_size

    @property
    def shelf_ids(self) -> chex.Array:
        return self._shelf_ids

    @property
    def not_in_queue_size(self) -> chex.Array:
        return self._not_in_queue_size

    @property
    def highways(self) -> chex.Array:
        return self._highways

    @property
    def goals(self) -> chex.Array:
        return self._goals


class RandomGenerator(GeneratorBase):
    """Randomly generates `RobotWarehouse` environment state. This generator places agents at
    starting positions on the grid and selects the requested shelves uniformly at random.
    """

    def __init__(
        self,
        shelf_rows: int,
        shelf_columns: int,
        column_height: int,
        num_agents: int,
        sensor_range: int,
        request_queue_size: int,
    ) -> None:
        """Initialises an robot_warehouse generator, used to generate grids for
        the RobotWarehouse environment."""
        super().__init__(
            shelf_rows,
            shelf_columns,
            column_height,
            num_agents,
            sensor_range,
            request_queue_size,
        )

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `RobotWarehouse` state that contains the grid and the agents/shelves layout.

        Returns:
            A `RobotWarehouse` state.
        """
        # empty grid array
        grid = jnp.zeros((2, *self._grid_size), dtype=jnp.int32)

        # spawn random agents with random request queue
        key, agents, shelves, shelf_request_queue = spawn_random_entities(
            key,
            self._grid_size,
            self._agent_ids,
            self._shelf_ids,
            self._shelf_positions,
            self._request_queue_size,
        )
        grid = place_entities_on_grid(grid, agents, shelves)

        # compute action mask
        action_mask = compute_action_mask(grid, agents)

        # create environment state
        state = State(
            grid=grid,
            agents=agents,
            shelves=shelves,
            request_queue=shelf_request_queue,
            step_count=jnp.array(0, int),
            action_mask=action_mask,
            key=key,
        )

        return state
