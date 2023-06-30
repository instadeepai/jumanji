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

from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.pacman.types import State


# flake8: noqa: C901
def convert_maze_to_numpy() -> Any:
    """Generates a numpy maze from ascii"""
    ascii_maze = [
        "XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
        "X  S         XX         S  X",
        "X XXXX XXXXX XX XXXXX XXXX X",
        "X XXXXOXXXXX XX XXXXXOXXXX X",
        "X XXXX XXXXX XX XXXXX XXXX X",
        "X                          X",
        "X XXXX XX XXXXXXXX XX XXXX X",
        "X XXXX XX XXXXXXXX XX XXXX X",
        "X      XX   TXXT   XX      X",
        "XXXXXX XXXXX XX XXXXX XXXXXX",
        "XXXXXX XXXXX XX XXXXX XXXXXX",
        "XXXXXX XXT        TXX XXXXXX",
        "XXXXXX XX XXX XXXX XX XXXXXX",
        "XXXXXX XX X  G   X XX XXXXXX",
        "           GXXXXG           ",
        "XXXXXX XX X  G   X XX XXXXXX",
        "XXXXXX XX XXX XXXX XX XXXXXX",
        "XXXXXX XX          XX XXXXXX",
        "XXXXXX XX XXXXXXXX XX XXXXXX",
        "XXXXXX XX XXXXXXXX XX XXXXXX",
        "X            XX            X",
        "X XXXX XXXXX XX XXXXX XXXX X",
        "X XXXX XXXXX XX XXXXX XXXX X",
        "X   XX S      P    S  XX   X",
        "XXX XX XX XXXXXXXX XX XX XXX",
        "XXX XX XX XXXXXXXX XX XX XXX",
        "X      XX    XX    XX      X",
        "X XXXXXXXXXX XX XXXXXXXXXX X",
        "X XXXXXXXXXX XX XXXXXXXXXX X",
        "X   O                 O    X",
        "XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    ]

    numpy_maze = []
    cookie_spaces = []
    powerup_spaces = []
    reachable_spaces = []
    ghost_spawns = []
    init_targets = []
    scatter_targets = []
    player_coords = None

    for x, row in enumerate(ascii_maze):
        size = (len(row), x + 1)
        binary_row = []
        for y, column in enumerate(row):
            if column == "G":
                ghost_spawns.append((y, x))
            if column == "P":
                player_coords = (y, x)
            if column == "X":
                binary_row.append(0)
            else:
                binary_row.append(1)
                cookie_spaces.append((y, x))
                reachable_spaces.append((y, x))
                if column == "O":
                    powerup_spaces.append((y, x))
                if column == "T":
                    init_targets.append((y, x))
                if column == "S":
                    scatter_targets.append((y, x))

        numpy_maze.append(binary_row)

    return (
        numpy_maze,
        cookie_spaces,
        powerup_spaces,
        reachable_spaces,
        ghost_spawns,
        player_coords,
        init_targets,
        scatter_targets,
    )


def create_grid_image(state: State) -> chex.Array:
    """
    Generate the observation of the current state.

    Args:
        state: 'State` object corresponding to the new state of the environment.

    Returns:
        rgb: A 3-dimensional array representing the RGB observation of the current state.
    """

    layer_1 = jnp.array(state.grid) * 0.66
    layer_2 = jnp.array(state.grid) * 0.0
    layer_3 = jnp.array(state.grid) * 0.33
    player_loc = state.player_locations
    ghost_pos = state.ghost_locations
    pellets_loc = state.power_up_locations
    is_scared = state.frightened_state_time
    idx = state.fruit_locations

    # Pellets are light orange
    for i in range(len(idx)):
        if jnp.array(idx[i]).sum != 0:
            loc = idx[i]
            layer_3 = layer_3.at[loc[1], loc[0]].set(1)
            layer_2 = layer_2.at[loc[1], loc[0]].set(0.8)
            layer_1 = layer_1.at[loc[1], loc[0]].set(0.6)

    # Power pellet is purple
    # print(pellets_loc)
    for i in range(len(pellets_loc)):
        p = pellets_loc[i]
        layer_1 = layer_1.at[p[1], p[0]].set(0.5)
        layer_2 = layer_2.at[p[1], p[0]].set(0)
        layer_3 = layer_3.at[p[1], p[0]].set(0.5)

    # Set player is yellow
    layer_1 = layer_1.at[player_loc.x, player_loc.y].set(1)
    layer_2 = layer_2.at[player_loc.x, player_loc.y].set(1)
    layer_3 = layer_3.at[player_loc.x, player_loc.y].set(0)

    cr = jnp.array([1, 1, 0, 1])
    cg = jnp.array([0, 0.7, 1, 0.7])
    cb = jnp.array([0, 1, 1, 0.35])
    # Set ghost locations

    layers = (layer_1, layer_2, layer_3)

    def set_ghost_colours(
        layers: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        layer_1, layer_2, layer_3 = layers
        for i in range(4):
            y = ghost_pos[i][0]
            x = ghost_pos[i][1]
            layer_1 = layer_1.at[x, y].set(cr[i])
            layer_2 = layer_2.at[x, y].set(cg[i])
            layer_3 = layer_3.at[x, y].set(cb[i])
        return layer_1, layer_2, layer_3

    def set_ghost_colours_scared(
        layers: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        layer_1, layer_2, layer_3 = layers
        for i in range(4):
            y = ghost_pos[i][0]
            x = ghost_pos[i][1]
            layer_1 = layer_1.at[x, y].set(0)
            layer_2 = layer_2.at[x, y].set(0)
            layer_3 = layer_3.at[x, y].set(1)
        return layer_1, layer_2, layer_3

    layers = jax.lax.cond(
        is_scared > 0, set_ghost_colours_scared, set_ghost_colours, layers
    )
    layer_1, layer_2, layer_3 = layers

    obs = [layer_1, layer_2, layer_3]
    rgb = jnp.stack(obs, axis=-1)

    return rgb
