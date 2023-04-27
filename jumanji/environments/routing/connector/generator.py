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
import jax
import jax.numpy as jnp

from jumanji.environments.routing.connector.generation_methods import ParallelRandomWalk
from jumanji.environments.routing.connector.types import Agent, State
from jumanji.environments.routing.connector.utils import get_position, get_target


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


class UniformRandomGenerator(Generator):
    """Randomly generates `Connector` grids that may or may not be solvable. This generator places
    start and target positions uniformly at random on the grid.
    """

    def __init__(self, grid_size: int, num_agents: int) -> None:
        """Instantiates a `UniformRandomGenerator`.

        Args:
            grid_size: size of the square grid to generate.
            num_agents: number of agents/paths on the grid.
        """
        super().__init__(grid_size, num_agents)

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `Connector` state that contains the grid and the agents' layout.

        Returns:
            A `Connector` state.
        """
        key, pos_key = jax.random.split(key)
        starts_flat, targets_flat = jax.random.choice(
            key=pos_key,
            a=jnp.arange(self.grid_size**2),
            shape=(2, self.num_agents),  # Start and target positions for all agents
            replace=False,  # Start and target positions cannot overlap
        )

        # Create 2D points from the flat arrays.
        starts = jnp.divmod(starts_flat, self.grid_size)
        targets = jnp.divmod(targets_flat, self.grid_size)

        # Get the agent values for starts and positions.
        agent_position_values = jax.vmap(get_position)(jnp.arange(self.num_agents))
        agent_target_values = jax.vmap(get_target)(jnp.arange(self.num_agents))

        # Create empty grid.
        grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)

        # Place the agent values at starts and targets.
        grid = grid.at[starts].set(agent_position_values)
        grid = grid.at[targets].set(agent_target_values)

        # Create the agent pytree that corresponds to the grid.
        agents = jax.vmap(Agent)(
            id=jnp.arange(self.num_agents),
            start=jnp.stack(starts, axis=1),
            target=jnp.stack(targets, axis=1),
            position=jnp.stack(starts, axis=1),
        )

        step_count = jnp.array(0, jnp.int32)

        return State(key=key, grid=grid, step_count=step_count, agents=agents)


class ParallelRandomWalkGenerator(Generator):
    """Randomly generates `Connector` grids that may or may not be solvable. This generator places
    start and target positions uniformly at random on the grid.
    """

    def __init__(self, grid_size: int, num_agents: int) -> None:
        """Instantiates a `UniformRandomGenerator`.

        Args:
            grid_size: size of the square grid to generate.
            num_agents: number of agents/paths on the grid.
        """
        super().__init__(grid_size, num_agents)
        self.board_generator = ParallelRandomWalk(
            self.grid_size, self.grid_size, self.num_agents
        )

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `Connector` state that contains the grid and the agents' layout.

        Returns:
            A `Connector` state.
        """
        key, pos_key = jax.random.split(key)

        grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
        starts, targets, solved_grid = self.board_generator.generate_board(key)
        starts = tuple(starts)
        targets = tuple(targets)
        agent_position_values = jax.vmap(get_position)(jnp.arange(self.num_agents))
        agent_target_values = jax.vmap(get_target)(jnp.arange(self.num_agents))

        # Transpose the agent_position_values to match the shape of the grid.
        # Place the agent values at starts and targets.
        grid = grid.at[starts].set(agent_position_values)
        grid = grid.at[targets].set(agent_target_values)

        # Create the agent pytree that corresponds to the grid.

        agents = jax.vmap(Agent)(
            id=jnp.arange(self.num_agents),
            start=jnp.stack(starts, axis=1),
            target=jnp.stack(targets, axis=1),
            position=jnp.stack(starts, axis=1),
        )

        step_count = jnp.array(0, jnp.int32)

        return State(key=key, grid=grid, step_count=step_count, agents=agents)
