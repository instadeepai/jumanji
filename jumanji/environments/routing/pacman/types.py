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

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex


class Position(NamedTuple):
    x: int
    y: int


class GhostPositions(NamedTuple):
    x: chex.Array
    y: chex.Array


@dataclass
class State:
    """
    key: random key used to set random actions.
    grid: a 2D array for the pacman maze.
    pellets: the number of remaining pellets in the game.
    frightened_state: a boolean indicating whether the frightened state has been triggered.
    frightened_state_time: the number of steps left for the frightened state.
    fruit_locations: the locations of the fruits in the game.
    power_up_locations: the locations of the power-ups in the game.
    player_locations: the location of the player.
    ghost_locations: the locations of the ghosts.
    initial_player_locations: the initial location of the player.
    initial_ghost_positions: the initial locations of the ghosts.
    last_direction: the last direction the player moved in.
    dead: a boolean indicating whether the player has collided with a ghost.
    lives: the number of lives the player has remaining.
    visited_index: a numpy array representing the indices visited by the player.
    """

    key: chex.PRNGKey
    grid: chex.Array  # maze
    pellets: int  # remaining pellets in level
    frightened_state: int  # is frightened state triggered
    frightened_state_time: int  # number of steps left for frightened state
    fruit_locations: chex.Array
    power_up_locations: chex.Array
    player_locations: Position
    ghost_locations: chex.Array
    initial_player_locations: Position
    initial_ghost_positions: chex.Array
    ghost_init_targets: chex.Array
    old_ghost_locations: chex.Array
    ghost_init_steps: chex.Array
    ghost_actions: chex.Array
    last_direction: int
    dead: bool
    visited_index: chex.Array
    ghost_starts: chex.Array
    scatter_targets: chex.Array


class Observation(NamedTuple):
    """
    grid: a 3D array that includes locations of the player, ghosts and items.
    """

    grid: chex.Array  # (31, 28, 3)
