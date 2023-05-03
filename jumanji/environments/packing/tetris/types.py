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

from typing import TYPE_CHECKING, NamedTuple

import chex

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class State:
    """
    grid_padded: the game grid, filled with zeros for the empty cells
        and with positive values for the filled cells. To allow for the placement of tetrominoes
        at the extreme right or bottom of the grid, the array has a padding of 3 columns on
        the right and 3 rows at the bottom. This padding enables the encoding of tetrominoes
        as 4x4 matrices, while ensuring that they can be placed without going out of bounds.
    grid_padded_old: similar to grid padded, used to save the grid before
        placing the last tetrominoe.
    tetrominoe_index: index to map the tetrominoe block.
    old_tetrominoe_rotated: a copy of the placed tetrominoe in the last step.
    new_tetrominoe: the new tetrominoe that needs to be placed.
    x_position: the selected x position for the last placed tetrominoe.
    y_position: the calculated y position for the last placed tetrominoe.
    action_mask: array of booleans that indicate the feasible actions, i.e. valid
        directions to move in.
    full_lines: saves the full lines in the last step.
    score: cumulative reward
    reward: instant reward
    key: random key used to generate random numbers at each step and for auto-reset.
    is_reset: True if the state is generated from a reset.
    """

    grid_padded: chex.Array  # (num_rows+3, num_cols+3)
    grid_padded_old: chex.Array  # (num_rows+3, num_cols+3)
    tetrominoe_index: chex.Numeric  # index for selecting a tetrominoe
    old_tetrominoe_rotated: chex.Array  # (4, 4)
    new_tetrominoe: chex.Array  # (4, 4)
    x_position: chex.Array  # (1,)
    y_position: chex.Array  # (1,)
    action_mask: chex.Array  # (4, num_cols)
    full_lines: chex.Array  # (num_rows)
    score: chex.Array  # (1,)
    reward: chex.Array  # (1,)
    key: chex.PRNGKey  # (2,)
    is_reset: chex.Array  # (1)


class Observation(NamedTuple):
    """
    grid: the game grid, filled with zeros for the empty cells and with
            ones for the filled cells.
    tetrominoe: matrix of size (4x4) of booleans (True for filled cells and False for empty cells).
    action_mask: array of booleans that indicate the feasible actions, i.e. valid
            directions to move in.
    """

    grid: chex.Array  # (num_rows, num_cols)
    tetrominoe: chex.Array  # (4, 4)
    action_mask: chex.Array  # (4, num_cols)
