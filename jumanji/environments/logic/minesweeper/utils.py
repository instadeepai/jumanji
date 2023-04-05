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
import jax
import jax.numpy as jnp

from jumanji.environments.logic.minesweeper.constants import (
    IS_MINE,
    PATCH_SIZE,
    UNEXPLORED_ID,
)
from jumanji.environments.logic.minesweeper.types import State


def create_flat_mine_locations(
    key: chex.PRNGKey,
    num_rows: int,
    num_cols: int,
    num_mines: int,
) -> chex.Array:
    """Create locations of mines on a board with a specified height, width, and number
    of mines. The locations are in flattened coordinates.

    Args:
        key: used for sampling mine positions.
        num_rows: the height of the board.
        num_cols: the width of the board.
        num_mines: how many mines to place.
    """
    return jax.random.choice(
        key,
        num_rows * num_cols,
        shape=(num_mines,),
        replace=False,
    )


def is_solved(state: State) -> chex.Array:
    """Check if all non-mined squares have been explored."""
    board = state.board
    num_mines = state.flat_mine_locations.shape[-1]
    num_rows, num_cols = board.shape
    num_explored = (board >= 0).sum()
    return num_explored == num_rows * num_cols - num_mines


def is_valid_action(state: State, action: chex.Array) -> chex.Array:
    """Check if an action is exploring a square that has not already been explored."""
    return state.board[tuple(action)] == UNEXPLORED_ID


def get_mined_board(state: State) -> chex.Array:
    """Compute the board with 1 in mine locations, otherwise 0."""
    return (
        jnp.zeros((state.board.shape[-1] * state.board.shape[-2],), dtype=jnp.int32)
        .at[state.flat_mine_locations]
        .set(IS_MINE)
    )


def explored_mine(state: State, action: chex.Array) -> chex.Array:
    """Check if an action is exploring a square containing a mine."""
    row, col = action
    index = col + row * state.board.shape[-1]
    mined_board = get_mined_board(state=state)
    return mined_board[index] == IS_MINE


def count_adjacent_mines(state: State, action: chex.Array) -> chex.Array:
    """Count the number of mines in a 3x3 patch surrounding the selected action."""
    action_row, action_col = action
    mined_board = get_mined_board(state=state).reshape(*state.board.shape)
    pad_board = jnp.pad(mined_board, pad_width=PATCH_SIZE - 1)
    selected_rows = jax.lax.dynamic_slice_in_dim(
        pad_board, start_index=action_row + 1, slice_size=PATCH_SIZE, axis=-2
    )
    return (
        jax.lax.dynamic_slice_in_dim(
            selected_rows,
            start_index=action_col + 1,
            slice_size=PATCH_SIZE,
            axis=-1,
        ).sum()
        - mined_board[tuple(action)]
    )
