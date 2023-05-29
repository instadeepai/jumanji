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
from typing import Any, Tuple

import chex
import jax
from jax import numpy as jnp

from jumanji.environments.routing.connector.constants import (
    DOWN,
    EMPTY,
    LEFT,
    NOOP,
    PATH,
    POSITION,
    RIGHT,
    TARGET,
    UP,
)
from jumanji.environments.routing.connector.types import Agent, State
from jumanji.environments.routing.connector.utils import (
    get_agent_grid,
    get_correction_mask,
    get_position,
    get_target,
    move_agent,
    move_position,
)


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


class RandomWalkGenerator(Generator):
    """Randomly generates `Connector` grids that are guaranteed be solvable.

    This generator places start positions randomly on the grid and performs a random walk from each.
    Targets are placed at their terminuses.
    """

    def __init__(self, grid_size: int, num_agents: int) -> None:
        """Instantiates a `RandomWalkGenerator.

        Args:
            grid_size: size of the square grid to generate.
            num_agents: number of agents/paths on the grid.
        """
        super().__init__(grid_size, num_agents)

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `Connector` state that contains the grid and the agents' layout.

        Args:
            key: used to randomly generate the connector grid.

        Returns:
            A `Connector` state.
        """
        key, board_key = jax.random.split(key)
        solved_grid, agents, grid = self.generate_board(board_key)
        step_count = jnp.array(0, jnp.int32)

        return State(key=key, grid=grid, step_count=step_count, agents=agents)

    def generate_board(self, key: chex.PRNGKey) -> Tuple[chex.Array, Agent, chex.Array]:
        """Generates solvable board using random walk.

        Args:
            key: random key.

        Returns:
            Tuple containing solved board, the agents and an empty training board.
        """
        grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
        key, step_key = jax.random.split(key)
        grid, agents = self._initialize_agents(key, grid)

        stepping_tuple = (step_key, grid, agents)

        _, grid, agents = jax.lax.while_loop(
            self._continue_stepping, self._step, stepping_tuple
        )

        # Convert heads and targets to format accepted by generator
        heads = agents.start.T
        targets = agents.position.T

        solved_grid = self.update_solved_board_with_head_target_encodings(
            grid, tuple(heads), tuple(targets)
        )
        # Update agent information to include targets and positions after first step
        agents.target = agents.position
        agents.position = agents.start

        agent_position_values = get_position(jnp.arange(self.num_agents))
        agent_target_values = get_target(jnp.arange(self.num_agents))
        # Populate an empty grid with heads and targets
        grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
        grid = grid.at[tuple(agents.start.T)].set(agent_position_values)
        grid = grid.at[tuple(agents.target.T)].set(agent_target_values)
        return solved_grid, agents, grid

    def _step(
        self, stepping_tuple: Tuple[chex.PRNGKey, chex.Array, Agent]
    ) -> Tuple[chex.PRNGKey, chex.Array, Agent]:
        """Takes one step for all agents."""
        key, grid, agents = stepping_tuple
        key, next_key = jax.random.split(key)
        agents, grid = self._step_agents(key, grid, agents)
        return next_key, grid, agents

    def _step_agents(
        self, key: chex.PRNGKey, grid: chex.Array, agents: Agent
    ) -> Tuple[Agent, chex.Array]:
        """Steps all agents at the same time correcting for possible collisions.

        If a collision occurs we place the agent with the lower `agent_id` in its previous position.
        This method is equivalent in function to _step_agents from 'Connector' environment.

        Returns:
            Tuple of agents and grid after having applied each agents' action
        """
        agent_ids = jnp.arange(self.num_agents)
        keys = jax.random.split(key, num=self.num_agents)

        # Randomly select action for each agent
        actions = jax.vmap(self._select_action, in_axes=(0, None, 0))(
            keys, grid, agents
        )

        # Step all agents at the same time (separately) and return all of the grids
        new_agents, grids = jax.vmap(self._step_agent, in_axes=(0, None, 0))(
            agents, grid, actions
        )

        # Get grids with only values related to a single agent.
        # For example: remove all other agents from agent 1's grid. Do this for all agents.
        agent_grids = jax.vmap(get_agent_grid)(agent_ids, grids)
        joined_grid = jnp.max(agent_grids, 0)  # join the grids

        # Create a correction mask for possible collisions (see the docs of `get_correction_mask`)
        correction_fn = jax.vmap(get_correction_mask, in_axes=(None, None, 0))
        correction_masks, collided_agents = correction_fn(grid, joined_grid, agent_ids)
        correction_mask = jnp.sum(correction_masks, 0)

        # Correct state.agents
        # Get the correct agents, either old agents (if collision) or new agents if no collision
        agents = jax.vmap(
            lambda collided, old_agent, new_agent: jax.lax.cond(
                collided,
                lambda: old_agent,
                lambda: new_agent,
            )
        )(collided_agents, agents, new_agents)
        # Create the new grid by fixing old one with correction mask and adding the obstacles
        return agents, joined_grid + correction_mask

    def _initialize_agents(
        self, key: chex.PRNGKey, grid: chex.Array
    ) -> Tuple[chex.Array, Agent]:
        """Initializes agents using random starting point and places heads on the grid.

        Args:
            key: random key.
            grid: empty grid.

        Returns:
            Tuple of grid with populated starting points and agents initialized with
            the same starting points.
        """
        # Generate locations of heads and an adjacent first move for each agent.
        # Return a grid with these positions populated.
        carry, heads_and_positions = jax.lax.scan(
            self._initialize_starts_and_first_move,
            (key, grid.reshape(-1)),
            jnp.arange(self.num_agents),
        )
        starts_flat, first_move_flat = heads_and_positions
        key, grid = carry
        grid = grid.reshape((self.grid_size, self.grid_size)).astype(jnp.int32)

        # Create 2D points from the flat arrays.
        starts = jnp.divmod(starts_flat, self.grid_size)
        first_step = jnp.divmod(first_move_flat, self.grid_size)
        # Fill target with default value as targets will be assigned after random walk
        targets = jnp.full((2, self.num_agents), -1)

        # # Initialize agents
        agents = jax.vmap(Agent)(
            id=jnp.arange(self.num_agents),
            start=jnp.stack(starts, axis=1),
            target=jnp.stack(targets, axis=1),
            position=jnp.stack(first_step, axis=1),
        )
        return grid, agents

    def _initialize_starts_and_first_move(
        self,
        carry: Tuple[chex.PRNGKey, chex.Array],
        agent_id: int,
    ) -> Tuple[Tuple[chex.PRNGKey, chex.Array], Tuple[chex.Array, chex.Array]]:
        """Initializes the starting positions and firs move of each agent.

        Args:
            carry: contains the current state of the flattened grid and the random key.
            agent_id: id of the agent whose positions are looked for.

        Returns:
            Tuple of indices of the starting position and the first move (in flat coordinates).
        """
        key, flat_grid = carry
        key, next_key = jax.random.split(key)
        grid_mask = flat_grid == 0
        start_coordinate_flat = jax.random.choice(
            key=key,
            a=jnp.arange(self.grid_size**2),
            shape=(),
            replace=True,
            p=grid_mask,
        )
        flat_grid = flat_grid.at[start_coordinate_flat].set(get_target(agent_id))
        available_cells = self._available_cells(
            flat_grid.reshape((self.grid_size, self.grid_size)), start_coordinate_flat
        )
        first_move_coordinate_flat = jax.random.choice(
            key=key,
            a=available_cells,
            shape=(),
            replace=True,
            p=available_cells != -1,
        )
        flat_grid = flat_grid.at[first_move_coordinate_flat].set(get_position(agent_id))
        return (key, flat_grid), (start_coordinate_flat, first_move_coordinate_flat)

    def _place_agent_heads_on_grid(self, grid: chex.Array, agent: Agent) -> chex.Array:
        """Updates grid with agent starting positions."""
        return grid.at[tuple(agent.start)].set(get_position(agent.id))

    def _continue_stepping(
        self, stepping_tuple: Tuple[chex.PRNGKey, chex.Array, Agent]
    ) -> chex.Array:
        """Determines if agents can continue taking steps."""
        _, grid, agents = stepping_tuple
        dones = jax.vmap(self._no_available_cells, in_axes=(None, 0))(grid, agents)
        return ~dones.all()

    def _no_available_cells(self, grid: chex.Array, agent: Agent) -> chex.Array:
        """Checks if there are no moves are available for the agent."""
        cell = self._convert_tuple_to_flat_position(agent.position)
        return (self._available_cells(grid, cell) == -1).all()

    def _select_action(
        self, key: chex.PRNGKey, grid: chex.Array, agent: Agent
    ) -> chex.Array:
        """Selects action for agent to take given its current position.

        Args:
            key: random key.
            grid: current state of the grid.
            agent: the agent.

        Returns:
            Integer corresponding to the action the agent will take in its next step.
            Action indices match those in connector.constants.
        """
        cell = self._convert_tuple_to_flat_position(agent.position)
        available_cells = self._available_cells(grid=grid, cell=cell)
        step_coordinate_flat = jax.random.choice(
            key=key,
            a=available_cells,
            shape=(),
            replace=True,
            p=available_cells != -1,
        )

        action = self._action_from_positions(cell, step_coordinate_flat)
        return action

    def _convert_flat_position_to_tuple(self, position: chex.Array) -> chex.Array:
        return jnp.array(
            [(position // self.grid_size), (position % self.grid_size)], dtype=jnp.int32
        )

    def _convert_tuple_to_flat_position(self, position: chex.Array) -> chex.Array:
        return jnp.array((position[0] * self.grid_size + position[1]), jnp.int32)

    def _action_from_positions(
        self, position_1: chex.Array, position_2: chex.Array
    ) -> chex.Array:
        """Compares two positions and returns action id to get from one to the other."""
        position_1 = self._convert_flat_position_to_tuple(position_1)
        position_2 = self._convert_flat_position_to_tuple(position_2)
        action_tuple = position_2 - position_1
        return self._action_from_tuple(action_tuple)

    def _action_from_tuple(self, action_tuple: chex.Array) -> chex.Array:
        """Returns integer corresponding to taking action defined by action_tuple."""
        action_multiplier = jnp.array([UP, DOWN, LEFT, RIGHT, NOOP])
        actions = jnp.array(
            [
                (action_tuple == jnp.array([-1, 0])).all(axis=0),
                (action_tuple == jnp.array([1, 0])).all(axis=0),
                (action_tuple == jnp.array([0, -1])).all(axis=0),
                (action_tuple == jnp.array([0, 1])).all(axis=0),
                (action_tuple == jnp.array([0, 0])).all(axis=0),
            ]
        )
        actions = jnp.sum(actions * action_multiplier, axis=0)
        return actions

    def _adjacent_cells(self, cell: int) -> chex.Array:
        """Returns chex.Array of adjacent cells to the input.

        Given a cell, return a chex.Array of size 4 with the flat indices of
        adjacent cells. Padded with -1's if less than 4 adjacent cells (if on the edge of the grid).

        Args:
            cell: the flat index of the cell to find adjacent cells of.

        Returns:
            A chex.Array of size 4 with the flat indices of adjacent cells
            (padded with -1's if less than 4 adjacent cells).
        """
        available_moves = jnp.full(4, cell)
        direction_operations = jnp.array([-1 * self.grid_size, self.grid_size, -1, 1])
        # Create a mask to check 0 <= index < total size
        cells_to_check = available_moves + direction_operations
        is_id_in_grid = cells_to_check < self.grid_size * self.grid_size
        is_id_positive = 0 <= cells_to_check
        mask = is_id_positive & is_id_in_grid

        # Ensure adjacent cells doesn't involve going off the grid
        unflatten_available = jnp.divmod(cells_to_check, self.grid_size)
        unflatten_current = jnp.divmod(cell, self.grid_size)
        is_same_row = unflatten_available[0] == unflatten_current[0]
        is_same_col = unflatten_available[1] == unflatten_current[1]
        row_col_mask = is_same_row | is_same_col
        # Combine the two masks
        mask = mask & row_col_mask
        return jnp.where(mask, cells_to_check, -1)

    def _available_cells(self, grid: chex.Array, cell: chex.Array) -> chex.Array:
        """Returns list of cells that can be stepped into from the input cell's position.

        Given a cell and the grid of the board, see which adjacent cells are available to move to
        (i.e. are currently unoccupied) to avoid stepping over exisitng wires.

        Args:
            grid: the current layout of the board i.e. current grid.
            cell: the flat index of the cell to find adjacent cells of.

        Returns:
            A chex.Array of size 4 with the flat indices of adjacent cells.
        """
        adjacent_cells = self._adjacent_cells(cell)
        # Get the wire id of the current cell
        value = grid[jnp.divmod(cell, self.grid_size)]
        wire_id = (value - 1) // 3

        available_cells_mask = jax.vmap(self._is_cell_free, in_axes=(None, 0))(
            grid, adjacent_cells
        )
        # Also want to check if the cell is touching itself more than once
        touching_cells_mask = jax.vmap(
            self._is_cell_doubling_back, in_axes=(None, None, 0)
        )(grid, wire_id, adjacent_cells)
        available_cells_mask = available_cells_mask & touching_cells_mask
        available_cells = jnp.where(available_cells_mask, adjacent_cells, -1)
        return available_cells

    def _is_cell_free(
        self,
        grid: chex.Array,
        cell: chex.Array,
    ) -> chex.Array:
        """Check if a given cell is free, i.e. has a value of 0.

        Args:
            grid: the current grid of the board.
            cell: the flat index of the cell to check.

        Returns:
            Boolean indicating whether the cell is free or not.
        """
        coordinate = jnp.divmod(cell, self.grid_size)
        return (cell != -1) & (grid[coordinate] == 0)

    def _is_cell_doubling_back(
        self,
        grid: chex.Array,
        wire_id: int,
        cell: int,
    ) -> chex.Array:
        """Checks if moving into an adjacent position results in a wire doubling back on itself.

        Check if the cell is touching any of the wire's own cells more than once.
        This means looking for surrounding cells of value 3 * wire_id + POSITION or
        3 * wire_id + PATH.
        """
        # Get the adjacent cells of the current cell
        adjacent_cells = self._adjacent_cells(cell)

        def is_cell_doubling_back_inner(
            grid: chex.Array, cell: chex.Array
        ) -> chex.Array:
            coordinate = jnp.divmod(cell, self.grid_size)
            cell_value = grid[tuple(coordinate)]
            touching_self = (
                (cell_value == 3 * wire_id + POSITION)
                | (cell_value == 3 * wire_id + PATH)
                | (cell_value == 3 * wire_id + TARGET)
            )
            return (cell != -1) & touching_self

        # Count the number of adjacent cells with the same wire id
        doubling_back_mask = jax.vmap(is_cell_doubling_back_inner, in_axes=(None, 0))(
            grid, adjacent_cells
        )
        # If the cell is touching itself more than once, return False
        return jnp.sum(doubling_back_mask) <= 1

    def _step_agent(
        self,
        agent: Agent,
        grid: chex.Array,
        action: int,
    ) -> Tuple[Agent, chex.Array]:
        """Moves the agent according to the given action if it is possible.

        This method is equivalent in function to _step_agent from 'Connector' environment.

        Returns:
            Tuple of (agent, grid) after having applied the given action.
        """
        new_pos = move_position(agent.position, action)

        new_agent, new_grid = jax.lax.cond(
            self._is_valid_position(grid, agent, new_pos) & (action != NOOP),
            move_agent,
            lambda *_: (agent, grid),
            agent,
            grid,
            new_pos,
        )
        return new_agent, new_grid

    def _is_valid_position(
        self,
        grid: chex.Array,
        agent: Agent,
        position: chex.Array,
    ) -> chex.Array:
        """Checks to see if the specified agent can move to `position`.

        This method is mirrors the use of to is_valid_position from the 'Connector' environment.

        Args:
            grid: the environment state's grid.
            agent: the agent.
            position: the new position for the agent in tuple format.

        Returns:
            bool: True if the agent moving to position is valid.
        """
        row, col = position
        grid_size = grid.shape[0]

        # Within the bounds of the grid
        in_bounds = (0 <= row) & (row < grid_size) & (0 <= col) & (col < grid_size)
        # Cell is not occupied
        open_cell = (grid[row, col] == EMPTY) | (grid[row, col] == get_target(agent.id))
        # Agent is not connected
        not_connected = ~agent.connected

        return in_bounds & open_cell & not_connected

    def update_solved_board_with_head_target_encodings(
        self,
        solved_grid: chex.Array,
        heads: Tuple[Any, ...],
        targets: Tuple[Any, ...],
    ) -> chex.Array:
        """Updates grid array with all agent encodings."""
        agent_position_values = get_position(jnp.arange(self.num_agents))
        agent_target_values = get_target(jnp.arange(self.num_agents))
        # Transpose the agent_position_values to match the shape of the grid.
        # Place the agent values at starts and targets.
        solved_grid = solved_grid.at[heads].set(agent_position_values)
        solved_grid = solved_grid.at[targets].set(agent_target_values)
        return solved_grid
