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

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.connector.constants import (
    DOWN,
    UP,
    LEFT,
    RIGHT,
    NOOP,
    POSITION,
    PATH,
    TARGET,
    EMPTY,
)
from jumanji.environments.routing.connector.types import Agent
from jumanji.environments.routing.connector.utils import (
    move_position,
    move_agent,
    get_agent_grid,
    get_correction_mask,
    get_target,
    get_position,
)


class ParallelRandomWalk:
    def __init__(self, rows: int, cols: int, num_agents: int):
        self.cols = cols
        self.rows = rows
        self.num_agents = num_agents

    def generate_board(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Generates solvable board using random walk.

        Args:
            key: random key.

        Returns:
            Tuple containing head and targets positions for each wire and the solved board
            generated in the random walk.
        """
        grid = self._return_blank_board()
        key, step_key = jax.random.split(key)
        grid, agents = self._initialise_agents(key, grid)

        stepping_tuple = (key, grid, agents)

        _, grid, agents = jax.lax.while_loop(
            self._continue_stepping, self._step, stepping_tuple
        )

        # Convert heads and targets to format accepted by generator
        heads = agents.start.T
        targets = agents.position.T

        return heads, targets, grid

    def _step(
        self, stepping_tuple: Tuple[chex.PRNGKey, chex.Array, Agent]
    ) -> Tuple[chex.PRNGKey, chex.Array, Agent]:
        """Takes one step for all agents."""
        key, grid, agents = stepping_tuple
        agents, grid = self._step_agents(key, grid, agents)
        key, next_key = jax.random.split(key)
        return next_key, grid, agents

    def _step_agents(
        self, key: chex.PRNGKey, grid: chex.Array, agents: Agent
    ) -> Tuple[Agent, chex.Array]:
        """Steps all agents at the same time correcting for possible collisions.

        If a collision occurs we place the agent with the lower `agent_id` in its previous position.
        This method is equivalent in function to _step_agents from 'Connector' environment.

        Returns:
            Tupleof agents and grid after having applied each agents' action
        """
        agent_ids = jnp.arange(self.num_agents)
        keys = jax.random.split(key, num=self.num_agents)

        # Randomly select action for each agent
        actions = jax.vmap(self._select_action, in_axes=(0, None, 0))(
            keys, grid, agents
        )

        # Step all agents at the same time (separately) and return all of the grids
        agents, grids = jax.vmap(self._step_agent, in_axes=(0, None, 0))(
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
        )(collided_agents, agents, agents)
        # Create the new grid by fixing old one with correction mask and adding the obstacles
        return agents, joined_grid + correction_mask

    def _initialise_agents(
        self, key: chex.PRNGKey, grid: chex.Array
    ) -> Tuple[chex.Array, Agent]:
        """Initialises agents using random starting point and places heads on grid.

        Args:
            key: random key.
            grid: empty grid.

        Returns:
            Tuple of grid with populated starting points and agents initialised with
            the same starting points.
        """
        starts_flat = jax.random.choice(
            key=key,
            a=jnp.arange(self.rows * self.cols),
            shape=(1, self.num_agents),
            # Start positions for all agents
            replace=False,
        )

        # Create 2D points from the flat arrays.
        starts = jnp.divmod(starts_flat[0], self.rows)
        # Fill target with default value as targets will be assigned aftert random walk
        targets = jnp.full((2, self.num_agents), -1)

        # Initialise agents
        agents = jax.vmap(Agent)(
            id=jnp.arange(self.num_agents),
            start=jnp.stack(starts, axis=1),
            target=jnp.stack(targets, axis=1),
            position=jnp.stack(starts, axis=1),
        )
        # Place agent heads on grid
        grid = jax.vmap(self._place_agent_heads_on_grid, in_axes=(None, 0))(
            grid, agents
        )
        grid = grid.max(axis=0)
        grid = jnp.array(grid, dtype=int)
        return grid, agents

    def _place_agent_heads_on_grid(self, grid: chex.Array, agent: Agent) -> chex.Array:
        """Updates grid with agent starting positions."""
        return grid.at[agent.start[0], agent.start[1]].set(get_position(agent.id))

    def _continue_stepping(
        self, stepping_tuple: Tuple[chex.PRNGKey, chex.Array, Agent]
    ) -> bool:
        """Determines if agents can continue taking steps."""
        key, grid, agents = stepping_tuple
        dones = jax.vmap(self._is_any_step_possible, in_axes=(None, 0))(grid, agents)
        return ~dones.all()

    def _is_any_step_possible(self, grid: chex.Array, agent: Agent) -> bool:
        """Checks if any moves are available for the agent."""
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
        return jnp.array([(position // self.cols), (position % self.cols)], dtype=int)

    def _convert_tuple_to_flat_position(self, position: chex.Array) -> chex.Array:
        return jnp.array((position[0] * self.cols + position[1]), int)

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
        direction_operations = jnp.array([-1 * self.rows, self.rows, -1, 1])
        # Create a mask to check 0 <= index < total size
        cells_to_check = available_moves + direction_operations
        is_id_in_grid = cells_to_check < self.rows * self.cols
        is_id_positive = 0 <= cells_to_check
        mask = is_id_positive & is_id_in_grid

        # Ensure adjacent cells doesn't involve going off the grid
        unflatten_available = jnp.divmod(cells_to_check, self.rows)
        unflatten_current = jnp.divmod(cell, self.rows)
        is_same_row = unflatten_available[0] == unflatten_current[0]
        is_same_col = unflatten_available[1] == unflatten_current[1]
        row_col_mask = is_same_row | is_same_col
        # Combine the two masks
        mask = mask & row_col_mask
        return jnp.where(mask == 0, -1, cells_to_check)

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
        value = grid[jnp.divmod(cell, self.rows)]
        wire_id = (value - 1) // 3

        _, available_cells_mask = jax.lax.scan(self._is_cell_free, grid, adjacent_cells)
        # Also want to check if the cell is touching itself more than once
        _, touching_cells_mask = jax.lax.scan(
            self._is_cell_doubling_back, (grid, wire_id), adjacent_cells
        )
        available_cells_mask = available_cells_mask & touching_cells_mask
        available_cells = jnp.where(available_cells_mask == 0, -1, adjacent_cells)

        return available_cells

    def _is_cell_free(
        self,
        grid: chex.Array,
        cell: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """Check if a given cell is free, i.e. has a value of 0.

        Args:
            grid: the current grid of the board.
            cell: the flat index of the cell to check.

        Returns:
            A tuple of the new grid and a boolean indicating whether the cell is free or not.
        """
        coordinate = jnp.divmod(cell, self.rows)
        return grid, jax.lax.select(
            cell == -1, False, grid[coordinate[0], coordinate[1]] == 0
        )

    def _is_cell_doubling_back(
        self,
        grid_wire_id: Tuple[chex.Array, int],
        cell: int,
    ) -> Tuple[Tuple[chex.Array, int], bool]:
        """Checks if moving into and adjacent position would result in a wire doubling back on itself.

        Check if the cell is touching any of the wire's own cells more than once.
        This means looking for surrounding cells of value 3 * wire_id + POSITION or
        3 * wire_id + PATH.
        """
        grid, wire_id = grid_wire_id
        # Get the adjacent cells of the current cell
        adjacent_cells = self._adjacent_cells(cell)

        def is_cell_touching_self_inner(
            grid: chex.Array, cell: chex.Array
        ) -> Tuple[chex.Array, chex.Array]:
            coordinate = jnp.divmod(cell, self.rows)
            cell_value = grid[coordinate[0], coordinate[1]]
            touching_self = jnp.logical_or(
                jnp.logical_or(
                    cell_value == 3 * wire_id + POSITION,
                    cell_value == 3 * wire_id + PATH,
                ),
                cell_value == 3 * wire_id + TARGET,
            )
            return grid, jnp.where(cell == -1, False, touching_self)

        # Count the number of adjacent cells with the same wire id
        _, touching_self_mask = jax.lax.scan(
            is_cell_touching_self_inner, grid, adjacent_cells
        )
        # If the cell is touching itself more than once, return False
        return (grid, wire_id), jnp.where(jnp.sum(touching_self_mask) > 1, False, True)

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

    def _return_blank_board(self) -> chex.Array:
        """Return empty grid of correct size."""
        return jnp.zeros((self.rows, self.cols), dtype=int)
