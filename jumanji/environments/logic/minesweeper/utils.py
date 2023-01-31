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
import jax.numpy as jnp
from jax import random
from jax.lax import dynamic_slice_in_dim

from jumanji.environments.logic.minesweeper.constants import (
    DEFAULT_BOARD_HEIGHT,
    DEFAULT_BOARD_WIDTH,
    DEFAULT_NUM_MINES,
    IS_MINE,
    PATCH_SIZE,
    UNEXPLORED_ID,
)
from jumanji.environments.logic.minesweeper.types import Board, State
from jumanji.types import Action


def create_flat_mine_locations(
    key: chex.PRNGKey,
    board_height: int = DEFAULT_BOARD_HEIGHT,
    board_width: int = DEFAULT_BOARD_WIDTH,
    num_mines: int = DEFAULT_NUM_MINES,
) -> Board:
    """Create locations of mines on a board with a specified height, width, and number
    of mines.
    Locations are in flattened coordinates."""
    return random.choice(
        key,
        board_height * board_width,
        shape=(num_mines,),
        replace=False,
    )


def is_solved(state: State) -> chex.Array:
    """Check if all non-mined squares have been explored"""
    board = state.board
    num_mines = state.flat_mine_locations.shape[-1]
    board_height, board_width = board.shape
    num_explored = (board >= 0).sum()
    return num_explored == board_height * board_width - num_mines


def is_valid_action(state: State, action: Action) -> chex.Array:
    """Check if an action is exploring a square that has not already been explored"""
    action_height, action_width = action
    return state.board[action_height, action_width] == UNEXPLORED_ID


def get_mined_board(state: State) -> chex.Array:
    """Compute the board with 1 in mine locations, otherwise 0"""
    return (
        jnp.zeros((state.board.shape[-1] * state.board.shape[-2],), dtype=jnp.int32)
        .at[state.flat_mine_locations]
        .set(IS_MINE)
    )


def explored_mine(state: State, action: Action) -> chex.Array:
    """Check if an action is exploring a square containing a mine"""
    height, width = action
    index = width + height * state.board.shape[-1]
    mined_board = get_mined_board(state=state)
    return mined_board[index] == IS_MINE


def count_adjacent_mines(state: State, action: Action) -> chex.Array:
    """Count the number of mines in a 3x3 patch surrounding the selected action"""
    action_height, action_width = action
    mined_board = get_mined_board(state=state).reshape(
        state.board.shape[-2], state.board.shape[-1]
    )
    pad_board = jnp.pad(mined_board, pad_width=PATCH_SIZE - 1)
    selected_rows = dynamic_slice_in_dim(
        pad_board, start_index=action_height + 1, slice_size=PATCH_SIZE, axis=-2
    )
    return (
        dynamic_slice_in_dim(
            selected_rows,
            start_index=action_width + 1,
            slice_size=PATCH_SIZE,
            axis=-1,
        ).sum()
        - mined_board[action_height, action_width]
    )
