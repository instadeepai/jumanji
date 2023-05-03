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


import chex

# check if jax is installed properly
import jax
import jax.numpy as jnp


def sample_tetrominoe_list(key: chex.PRNGKey, tetrominoes_list: chex.Array) -> tuple:
    """Sample a tetrominoe from defined list of tetrominoes.

    Args:
        key:PRINGkey to generate a random tetrominoe
        tetrominoes_list:jnp array of all tetrominoes to sample from.

    Returns:
        tetrominoe:jnp bool array of size (4,4) represents the selected tetrominoe
        tetrominoe_index:index of the tetrominoe in the tetrominoes_list
    """
    # Generate a reandom integer from 0 to len(tetrominoes_list)
    tetrominoe_index = jax.random.randint(key, (1,), 0, len(tetrominoes_list))
    all_rotations = tetrominoes_list[tetrominoe_index]
    all_rotations = jnp.squeeze(all_rotations)
    rotation_index = 0
    tetrominoe = all_rotations[rotation_index]
    return tetrominoe, tetrominoe_index


def tetrominoe_action_mask(
    grid_padded: chex.Array, tetrominoe: chex.Array
) -> chex.Array:
    """check all possible positions for one side of a tetrominoe.
    Steps:
        1) fix y=0
        2) create a list for possible x position by checking the first line in the tetrominoe (y=0)
        3) for each tetrominoe's x_position check if the tetrominoe
           is in the right padding and generate a boolean list.
        4) join the 2 lists to get a list for all valid positions and not in the right padding.
        5) return the resulted list.

    Args:
        grid_padded:grid_padded to check positions in it.
        tetrominoe:tetrominoe to check  for all possible positions in the grid_padded.

    Returns:
        action_mask:jnp bool array of size (1,num_cols) corresponds
        to all possible positions for one side of a tetrominoe in the grid_padded.
    """
    tetrominoe_mask = tetrominoe.at[1, :].set(tetrominoe[1, :] + tetrominoe[2, :])
    tetrominoe_mask = tetrominoe_mask.at[0, :].set(
        tetrominoe_mask[0, :] + tetrominoe_mask[1, :]
    )
    tetrominoe_mask = jnp.clip(tetrominoe_mask, a_max=1)
    num_cols = grid_padded.shape[1] - 3
    list_action_mask = jnp.array(
        [
            grid_padded.at[0:4, i : i + 4].add(tetrominoe_mask)[0:4, i : i + 4].max()
            < 2
            for i in range(0, num_cols)
        ]
    )
    right_cells = jnp.array(
        [
            grid_padded.at[0:4, i : i + 4].add(tetrominoe)[:, num_cols:].max() < 1
            for i in range(0, num_cols)
        ]
    )
    action_mask = jnp.logical_and(list_action_mask, right_cells)
    return action_mask


def place_tetrominoe(
    grid_padded: chex.Array, tetrominoe: chex.Array, x_position: int
) -> chex.Array:
    """Place a Tetrominoe in the Game Grid.

    This function takes in a "grid" container, a "tetrominoe" block,
    and an 'x_position' value, and calculates the block's y position and
    places it in the grid.

    Args:
        grid_padded: The container of the new tetrominoe.
        tetrominoe: The tetrominoe that that that will be placed in the grid_padded.
        x_position: The position x of the tetrominoe in the grid_padded.

    Returns:
        grid_padded: The updated grid_padded with the new tetrominoe.
        y_position: Position of the tetrominoe
    """
    # `possible_positions` is a list of booleans with a size of 21,
    # It represents all possible y positions for a tetrominoe in the grid_padded.
    # A tetrominoe's possible position is a position where the tetrominoe is not on top
    # of a filled cell.
    num_rows = grid_padded.shape[0] - 3
    grid_padded_cliped = jnp.clip(grid_padded, a_max=1)
    possible_positions = jnp.array(
        [
            (
                grid_padded_cliped
                + jax.lax.dynamic_update_slice(
                    jnp.zeros_like(grid_padded_cliped), tetrominoe, (i, x_position)
                )
            ).max()
            < 2
            for i in range(0, num_rows)
        ]
    )
    tetrominoe_padd = tetrominoe.sum(axis=1) > 0
    # calculate the number of rows padding if needed at the begining.
    # true for possible padding and false for non possible padding.
    tetrominoe_padd = jnp.logical_not(jnp.flip(tetrominoe_padd[1:]))
    possible_padding = jnp.logical_and(possible_positions[-3:], tetrominoe_padd)
    possible_positions = possible_positions.at[-3:].set(possible_padding)
    y_position = jax.lax.cond(
        possible_positions[:-3].sum() == possible_positions.shape[0] - 3,
        lambda possible_positions: (jnp.argmin(possible_positions)).astype(int),
        lambda possible_positions: (jnp.argmin(possible_positions) - 1).astype(int),
        possible_positions,
    )
    # Calculate the last possible position before the first non possible position
    # (last true before the first false).
    y_position = (jnp.argmin(possible_positions) - 1).astype(int)
    # Update the grid_padded.
    tetrominoe_color_id = grid_padded.max() + 1
    tetrominoe = tetrominoe * tetrominoe_color_id
    new_grid_padded = jax.lax.dynamic_update_slice(
        grid_padded, tetrominoe, (y_position, x_position)
    )
    # get the max of the old and the new grid_padded.
    grid_padded = jnp.maximum(grid_padded, new_grid_padded)
    return grid_padded, y_position


def clean_lines(grid_padded: chex.Array, full_lines: chex.Array) -> chex.Array:
    """clean full lines in the grid_padded.
    Steps:
        1) Sort the indices of full lines in a list called 'indices',
           placing the full lines at the top, followed by non-full lines.
        2) Reshape 'indices' to have the same shape as the grid_padded.
        3) Reorder the grid_padded using the 'indices' list.
        4) Update any full lines in the grid_padded by converting them from 1s to 0s.
        5) Return the updated grid_padded.

    Args:
        grid_padded:Container of the game.
        full_lines:jnp array of size(1, num_rows) contains a list of
        booleans that represents full lines (full lines are true, and False for uncomplete lines).

    Returns:
        grid_padded:updated grid_padded.
    """
    indices = jnp.argsort(
        jnp.logical_not(full_lines)
    )  # Indices contains the indices that would sort an array.
    indices = jnp.reshape(
        indices, (indices.shape[0], 1)
    )  # Rechape indices to be similar to the grid_padded
    grid_padded = jnp.take_along_axis(
        grid_padded, indices, axis=0
    )  # Sort the grid_padded using indicies
    # Convert the full lines to zeros.
    grid_padded = jax.lax.fori_loop(
        0,
        full_lines.sum(),
        lambda i, grid_padded: grid_padded.at[i].set(0),
        grid_padded,
    )
    return grid_padded
