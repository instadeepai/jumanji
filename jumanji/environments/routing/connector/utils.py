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

import functools
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d

from jumanji.environments.routing.connector.constants import (
    EMPTY,
    PATH,
    POSITION,
    TARGET,
)
from jumanji.environments.routing.connector.types import Agent


def get_path(agent_id: jnp.int32) -> jnp.int32:
    """Get the value used in the state to represent the path of the given agent."""
    return PATH + 3 * agent_id


def get_position(agent_id: jnp.int32) -> jnp.int32:
    """Get the value used in the state to represent the position of the given agent."""
    return POSITION + 3 * agent_id


def get_target(agent_id: jnp.int32) -> jnp.int32:
    """Get the value used in the state to represent the target of the given agent."""
    return TARGET + 3 * agent_id


def is_target(value: int) -> bool:
    """Returns: True if the value on the grid is used to represent a target, false otherwise."""
    return (value > 0) & ((value - TARGET) % 3 == 0)


def is_position(value: int) -> bool:
    """Returns: True if the value on the grid is used to represent a position, false otherwise."""
    return (value > 0) & ((value - POSITION) % 3 == 0)


def is_path(value: int) -> bool:
    """Returns: True if the value on the grid is used to represent a path, false otherwise."""
    return (value > 0) & ((value - PATH) % 3 == 0)


def get_agent_id(value: int) -> int:
    """Returns: The ID of an agent given it's target, path or position."""
    return 0 if value == 0 else (value - 1) // 3 + 1


def move_position(position: chex.Array, action: jnp.int32) -> chex.Array:
    """Use a position and an action to return a new position.

    Args:
        position: a position representing row and column.
        action: the action representing cardinal directions.
    Returns:
        The new position after the move.
    """
    row, col = position

    move_noop = lambda row, col: jnp.array([row, col], jnp.int32)
    move_left = lambda row, col: jnp.array([row, col - 1], jnp.int32)
    move_up = lambda row, col: jnp.array([row - 1, col], jnp.int32)
    move_right = lambda row, col: jnp.array([row, col + 1], jnp.int32)
    move_down = lambda row, col: jnp.array([row + 1, col], jnp.int32)

    return jax.lax.switch(action, [move_noop, move_up, move_right, move_down, move_left], row, col)


def move_agent(agent: Agent, grid: chex.Array, new_pos: chex.Array) -> Tuple[Agent, chex.Array]:
    """Moves `agent` to `new_pos` on `grid`. Sets `agent`'s position to `new_pos`.

    Returns:
        An agent and grid representing the agent at the new_pos.
    """
    grid = grid.at[tuple(new_pos)].set(get_position(agent.id))
    grid = grid.at[tuple(agent.position)].set(get_path(agent.id))

    new_agent = Agent(
        id=agent.id,
        start=agent.start,
        target=agent.target,
        position=jnp.array(new_pos),
    )
    return new_agent, grid


def is_valid_position(
    value_on_grid: int, agent: Agent, position: chex.Array, grid_size: int
) -> chex.Array:
    """Checks to see if the specified agent can move to `position`.

    Args:
        grid: the environment state's grid.
        agent: the agent.
        position: the new position for the agent.

    Returns:
        bool: True if the agent moving to position is valid.
    """
    row, col = position

    # Within the bounds of the grid
    in_bounds = (0 <= row) & (row < grid_size) & (0 <= col) & (col < grid_size)
    # Cell is not occupied
    open_cell = (value_on_grid == EMPTY) | (value_on_grid == get_target(agent.id))
    # Agent is not connected
    not_connected = ~agent.connected

    return in_bounds & open_cell & not_connected


def connected_or_blocked(agent: Agent, action_mask: chex.Array) -> chex.Array:
    """Returns: `True` if an agent is connected or blocked, `False` otherwise."""
    return agent.connected.all() | jnp.logical_not(action_mask[1:].any())


def get_agent_grid(agent_id: jnp.int32, grid: chex.Array) -> chex.Array:
    """Returns the grid with zeros everywhere except locations related to the desired agent:
    path, position, or target represented by 1, 2, 3 for the first agent, 4, 5, 6 for the
    second agent, etc."""
    position = get_position(agent_id)
    target = get_target(agent_id)
    path = get_path(agent_id)
    agent_head = (grid == position) * position
    agent_target = (grid == target) * target
    agent_path = (grid == path) * path
    return agent_head + agent_target + agent_path


def get_action_masks(agents: Agent, grid: chex.Array) -> chex.Array:
    """Gets the action mask for all agents"""
    num_agents = len(agents.id)  # N in shape comments
    # Don't check action 0 because no-op is always valid
    actions_to_check = jnp.arange(1, 5)
    num_total_actions = 5  # Or derive from your action space definition

    new_positions = jax.vmap(
        jax.vmap(move_position, (None, 0)),
        (0, None),
    )(agents.position, actions_to_check)
    # 2. Fetch grid values at all calculated `new_positions`.
    #    `grid` has shape (grid_dim1, grid_dim2, ...).
    #    `new_positions` has shape (N, A, pos_dims).
    #    We need to gather values from `grid` at each of the (N*A) locations.
    #    `jnp.moveaxis(...)` changes shape from (N, 4, pos_dims) to (pos_dims, N, 4)
    #    `grid_val_at_new_positions` will have shape (N, A).
    grid_val_at_new_positions = grid[tuple(jnp.moveaxis(new_positions, -1, 0))]

    # 3. Initialize action masks.
    #    Assuming 5 total actions (0: no-op, 1-4: checked actions).
    #    The no-op (action 0) is always valid.
    all_masks = jnp.ones((num_agents, num_total_actions), dtype=bool)

    # 4. Determine validity for the 'actions_to_check'.
    #    This uses a "double vmap" strategy:
    #    - The inner vmap (`vmapped_is_valid_over_actions`) maps over the A actions for one agent
    #    - The outer vmap iterates this inner function over all N agents.

    #    `is_valid_position` signature: (grid_val, agent_slice, new_pos_slice, grid_size_scalar)
    #    Inner vmap function (`vmapped_is_valid_over_actions`):
    #      - Takes inputs corresponding to ONE agent:
    #        1. grid_vals_for_agent (A,): Grid values for A potential new positions.
    #        2. single_agent_data (Agent Pytree slice): Data for that one agent.
    #        3. new_positions_for_agent (A, pos_dims): A potential new positions.
    #        4. grid_shape_val (scalar): Relevant grid dimension (broadcasted over A actions).
    #      - Returns: (A,) boolean array for that agent.
    vmapped_is_valid_over_actions = jax.vmap(
        is_valid_position,
        in_axes=(0, None, 0, None),
    )

    # Outer vmap:
    #  - Applies `vmapped_is_valid_over_actions` to each agent.
    #  - Inputs:
    #    1. `grid_val_at_new_positions` (N, A): Each (A,) slice goes to `grid_vals_for_agent`.
    #    2. `agents` (Agent Pytree with N leading dim): Each slice goes to `single_agent_data`.
    #    3. `new_positions` (N, A, pos_dims): Each slice goes to `new_positions_for_agent`.
    #    4. `grid.shape[0]` (scalar): Broadcasted to `grid_shape_val`.
    #  - Returns: (N, A) boolean array: `validity_for_checked_actions`.
    validity_for_checked_actions = jax.vmap(
        vmapped_is_valid_over_actions,
        in_axes=(0, 0, 0, None),
    )(grid_val_at_new_positions, agents, new_positions, grid.shape[0])

    # 5. Update the initialized masks with the calculated validities.
    #    `all_masks` is (N, num_total_actions).
    #    `actions_to_check` (e.g., [1,2,3,4]) specifies which columns to update.
    #    `validity_for_checked_actions` is (N, A).
    all_masks = all_masks.at[:, actions_to_check].set(validity_for_checked_actions)

    return all_masks


# TODO: tests
def is_repeated_later(positions: chex.Array) -> chex.Array:
    """Creates a boolean mask for a 2D array of (x,y) positions in which True at an index means the
    (x,y) pair at that index apears later in the array

    Args:
      arr: A 2D array of shape (N, 2), where N is the number of (x,y) pairs.

    Returns:
      A 1D array of booleans with shape (N,).
    """
    n_agents = positions.shape[0]  # Number of (x,y) pairs

    # Validate shape
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            f"Input array must be 2-dimensional with shape (N, 2) for N > 0, "
            f"but got shape {positions.shape}."
        )

    # Step 1: Create a matrix where (i, j) is True if arr[i] (pair) == arr[j] (pair)
    # arr[:, None, :] reshapes arr from (N, 2) to (N, 1, 2)
    # arr[None, :, :] reshapes arr from (N, 2) to (1, N, 2)
    # Broadcasting these leads to element-wise comparison, resulting in shape (N, N, 2)
    # where the last dimension holds the comparison result for x and y separately.
    # The jnp.all(..., axis=-1) reduces along the last dimension to check if the pairs are equal.
    is_equal_pair = jnp.all(positions[:, None, :] == positions[None, :, :], axis=-1)

    # TODO: might be best to pre-compute and store this
    # Step 2: Create a mask where (i, j) is True if j > i (j is an index after i)
    indices = jnp.arange(n_agents)
    is_next = indices[None, :] > indices[:, None]  # Shape (N, N)

    # Step 3: Combine masks: True if arr[i] == arr[j] (as pairs) AND j > i then
    # check if any such preceding identical pair j exists.
    return jnp.any(is_equal_pair & is_next, axis=1)


def get_surrounded_mask(grid: chex.Array) -> chex.Array:
    """
    Args:
        grid: 2D JAX array where 0 = open cell, positive integers = occupied

    Returns:
        Boolean mask where True indicates a cell is surrounded by occupied cells
    """

    # Create binary occupancy map
    occupied = (grid > 0).astype(jnp.float32)

    # Kernel to check all 4 neighbors
    kernel = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    # Count occupied neighbors
    neighbor_count = convolve2d(occupied, kernel, mode="same")

    # Cell is surrounded if all possible neighbors are occupied
    possible_max_neighbors = convolve2d(jnp.ones_like(occupied), kernel, mode="same")
    surrounded = neighbor_count == possible_max_neighbors

    return surrounded


@functools.partial(jax.jit, static_argnums=(0,))
def get_adjacency_mask(grid_shape: tuple, coordinate: jax.Array) -> jax.Array:
    """
    Creates a mask with 1s on cells adjacent to a coordinate.

    Args:
        grid_shape: A tuple (rows, cols) defining the grid dimensions.
        coordinate: A JAX array or tuple (row, col) for the center point.

    Returns:
        A JAX array of shape `grid_shape` with 1s for adjacent cells
        and 0s elsewhere.
    """
    # 1. Start with a grid of zeros
    mask = jnp.zeros(grid_shape, dtype=jnp.int32)
    row, col = coordinate

    # 2. Define the four potential neighbor coordinates (N, S, E, W)
    neighbors = jnp.array(
        [
            [row - 1, col],  # North
            [row + 1, col],  # South
            [row, col + 1],  # East
            [row, col - 1],  # West
        ]
    )
    rows, cols = grid_shape
    valid = (
        (neighbors[:, 0] >= 0)
        & (neighbors[:, 0] < rows)
        & (neighbors[:, 1] >= 0)
        & (neighbors[:, 1] < cols)
    )

    # 3. Set the valid locations to 1, drop the invalid locations
    final_mask = mask.at[neighbors[:, 0], neighbors[:, 1]].set(valid)

    return final_mask
