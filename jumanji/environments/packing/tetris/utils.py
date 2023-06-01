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


def sample_tetromino_list(
    key: chex.PRNGKey, tetrominoes_list: chex.Array
) -> Tuple[chex.Array, chex.Array]:
    """Sample a tetromino from defined list of tetrominoes.

    Args:
        key:PRINGkey to generate a random tetrominoe
        tetrominoes_list:jnp array of all tetrominoes to sample from.

    Returns:
        tetromino:jnp bool array of size (4,4) represents the selected tetromino
        tetromino_index: index of the tetromino in the `tetrominoes_list`
    """
    # Generate a reandom integer from 0 to len(tetrominoes_list)
    tetromino_index = jax.random.randint(key, (), 0, len(tetrominoes_list))
    all_rotations = tetrominoes_list[tetromino_index]
    rotation_index = 0
    tetromino = all_rotations[rotation_index]
    return tetromino, tetromino_index


def check_valid_tetromino_placement(
    grid: chex.Array, tetromino: chex.Array, y_position: int, x_position: int
) -> chex.Array:
    """Determines whether a `tetromino` can be positioned within the `grid` at the specified
    `x_position` and `y_position` coordinates without overlapping any filled cells.

    Args:
        grid: the container of the game.
        tetromino: the tetromino that that that will be placed in the grid.
        x_position: the position x of the tetromino in the grid.
        y_position: the position y of the tetromino in the grid.

    Returns:
        chex.array of shape ().
    """
    crop = jax.lax.dynamic_slice(
        grid, start_indices=(y_position, x_position), slice_sizes=(4, 4)
    )
    crop = crop + tetromino
    return ~jnp.any(crop >= 2)


def tetromino_action_mask(grid_padded: chex.Array, tetromino: chex.Array) -> chex.Array:
    """Check all possible positions for one side of a `tetromino`.
    Steps:
        1) fix y=0
        2) create a list for possible x position by checking the first line in the tetromino (y=0)
        3) for each tetromino's x_position check if the tetromino
           is in the right padding and generate a boolean list.
        4) join the 2 lists to get a list for all valid positions and not in the right padding.
        5) return the resulted list.

    Args:
        grid_padded: grid_padded to check positions in it.
        tetromino: tetromino to check  for all possible positions in the `grid_padded`.

    Returns:
        action_mask: jnp bool array of size (num_cols,) corresponds
        to all possible positions for one side of a tetromino in the `grid_padded`.
    """
    tetromino_mask = tetromino.at[1, :].set(tetromino[1, :] + tetromino[2, :])
    tetromino_mask = tetromino_mask.at[0, :].set(
        tetromino_mask[0, :] + tetromino_mask[1, :]
    )
    tetromino_mask = jnp.clip(tetromino_mask, a_max=1)
    num_cols = grid_padded.shape[1] - 3
    # Check if tetromino can be placed at the top of the grid, if so it means
    # that the tetromino is placeble in this `x_position`.
    # The top of the grid may be an overhang where the grid is filled
    # from the top and empty in the middle.
    # To overcome this, use the tetromino mask instead of the tetromino, wich fills
    # the possible top cells to overcome any overhang position.
    list_action_mask = jax.vmap(
        check_valid_tetromino_placement,
        in_axes=(None, None, None, 0),
    )(grid_padded, tetromino_mask, 0, jnp.arange(num_cols))

    tetromino_padd = tetromino.sum(axis=0) > 0
    # Calculate the number of rows padding if needed at the begining.
    # True for possible padding and False for non possible padding.
    tetromino_padd = jnp.logical_not(jnp.flip(tetromino_padd[1:]))
    possible_padding = jnp.logical_and(list_action_mask[-3:], tetromino_padd)
    list_action_mask = list_action_mask.at[-3:].set(possible_padding)
    return list_action_mask


def place_tetromino(
    grid_padded: chex.Array, tetromino: chex.Array, x_position: int
) -> Tuple[chex.Array, int]:
    """Place a `tetromino` in the Game Grid.

    This function takes in a `grid` container, a `tetromino` block,
    and an `x_position` value, and calculates the block's y position and
    places it in the `grid`.

    Args:
        grid_padded: the container of the new `tetromino`.
        tetromino: the `tetromino` that that that will be placed in the `grid_padded`.
        x_position: the position x of the `tetromino` in the `grid_padded`.

    Returns:
        grid_padded: the updated `grid_padded` with the new `tetromino`.
        y_position: position of the `tetromino`
    """
    # `possible_positions` is a list of booleans with a size of 21,
    # It represents all possible y positions for a tetromino in the `grid_padded`.
    # A tetromino's possible position is a position where the tetromino is not on top
    # of a filled cell.
    num_rows = grid_padded.shape[0] - 3
    grid_padded_cliped = jnp.clip(grid_padded, a_max=1)
    # Check all possible y postions relative to the selected `x_position`
    # possible positions is a `chex.array` of shape (num_rows), contains True if a tetromino
    # can be placed in `y_position` without overlaping any filled cell.
    possible_positions = jax.vmap(
        check_valid_tetromino_placement,
        in_axes=(None, None, 0, None),
    )(grid_padded_cliped, tetromino, jnp.arange(num_rows), x_position)
    tetromino_padd = tetromino.sum(axis=1) > 0
    # Calculate the number of rows padding if needed at the begining.
    # True for possible padding and False for non possible padding.
    tetromino_padd = jnp.logical_not(jnp.flip(tetromino_padd[1:]))
    possible_padding = jnp.logical_and(possible_positions[-3:], tetromino_padd)
    possible_positions = possible_positions.at[-3:].set(possible_padding)
    y_position = jax.lax.cond(
        possible_positions[:-3].sum() == possible_positions.shape[0] - 3,
        lambda possible_positions: jnp.argmin(possible_positions),
        lambda possible_positions: jnp.argmin(possible_positions) - 1,
        possible_positions,
    )
    # Calculate the last possible position before the first non possible position
    # (last true before the first false).
    y_position = jnp.argmin(possible_positions) - 1
    # Update the `grid_padded`.
    tetromino_color_id = grid_padded.max() + 1
    tetromino = tetromino * tetromino_color_id
    new_grid_padded = jax.lax.dynamic_update_slice(
        grid_padded, tetromino, (y_position, x_position)
    )
    # Get the max of the old and the new `grid_padded`.
    grid_padded = jnp.maximum(grid_padded, new_grid_padded)
    return grid_padded, y_position


def clean_lines(grid_padded: chex.Array, full_lines: chex.Array) -> chex.Array:
    """Clean full lines in the `grid_padded`.
    Steps:
        1) Sort the `indices` of full lines in a list called `indices`,
           placing the full lines at the top, followed by non-full lines.
        2) Reshape 'indices' to have the same shape as the `grid_padded`.
        3) Reorder the `grid_padded` using the 'indices' list.
        4) Update any full lines in the `grid_padded` by converting them from 1s to 0s.
        5) Return the updated `grid_padded`.

    Args:
        grid_padded: container of the game.
        full_lines: jnp array of size(num_rows,) contains a list of
        booleans that represents full lines (full lines are true, and False for uncomplete lines).

    Returns:
        grid_padded: updated grid_padded.
    """
    indices = jnp.argsort(
        jnp.logical_not(full_lines)
    )  # `indices` contains the indices that would sort an array.
    indices = jnp.reshape(
        indices, (indices.shape[0], 1)
    )  # Rechape `indices` to be similar to the `grid_padded`
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
