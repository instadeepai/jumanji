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
import jax.numpy as jnp


class Position(NamedTuple):
    x: jnp.int32
    y: jnp.int32

    def __eq__(self, other: object) -> chex.Array:
        if not isinstance(other, Position):
            return NotImplemented
        return (self.x == other.x) & (self.y == other.y)


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
    pellets: jnp.int32  # remaining pellets in level
    frightened_state: jnp.int32  # is frightened state triggered
    frightened_state_time: jnp.int32  # number of steps left for frightened state
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
    last_direction: jnp.int32
    dead: jnp.bool_
    visited_index: chex.Array
    ghost_starts: chex.Array
    scatter_targets: chex.Array
    step_count: jnp.int32  # ()


class Observation(NamedTuple):
    """
    grid: a 3D array that includes locations of the player, ghosts and items.
    """

    grid: chex.Array  # (31, 28, 3)
    player_locations: Position
    ghost_locations: chex.Array
    power_up_locations: chex.Array
    frightened_state_time: jnp.int32
    fruit_locations: chex.Array
    action_mask: chex.Array
