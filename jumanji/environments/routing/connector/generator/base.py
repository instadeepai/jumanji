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

import chex

from jumanji.environments.routing.connector.types import State


class Generator(abc.ABC):
    """Base class for generators for the connector environment."""

    def __init__(self, grid_size: int, num_agents: int) -> None:
        """Initialises a connector generator, used to generate grids for the Connector environment.

        Args:
            grid_size: size of the grid to generate.
            num_agents: number of agents on the grid.
        """
        self._grid_size = grid_size
        self._num_agents = num_agents

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `Connector` state that contains the grid and the agents' layout.

        Returns:
            A `Connector` state.
        """
