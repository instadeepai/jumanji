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

from typing import Tuple, Union

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.pacman.types import Observation, State


# flake8: noqa: C901
def create_grid_image(observation: Union[Observation, State]) -> chex.Array:
    """
    Generate the observation of the current state.

    Args:
        state: 'State` object corresponding to the new state of the environment.

    Returns:
        rgb: A 3-dimensional array representing the RGB observation of the current state.
    """

    # Make walls blue and passages black
    layer_1 = (1 - observation.grid) * 0.0
    layer_2 = (1 - observation.grid) * 0.0
    layer_3 = (1 - observation.grid) * 0.6

    player_loc = observation.player_locations
    ghost_pos = observation.ghost_locations
    pellets_loc = observation.power_up_locations
    is_scared = observation.frightened_state_time
    idx = observation.pellet_locations
    n = 3

    # Power pellet are pink
    for i in range(len(pellets_loc)):
        p = pellets_loc[i]
        layer_1 = layer_1.at[p[1], p[0]].set(1.0)
        layer_2 = layer_2.at[p[1], p[0]].set(0.8)
        layer_3 = layer_3.at[p[1], p[0]].set(0.6)

    # Set player is yellow
    layer_1 = layer_1.at[player_loc.x, player_loc.y].set(1)
    layer_2 = layer_2.at[player_loc.x, player_loc.y].set(1)
    layer_3 = layer_3.at[player_loc.x, player_loc.y].set(0)

    cr = jnp.array([1, 1, 0, 1])
    cg = jnp.array([0, 0.7, 1, 0.5])
    cb = jnp.array([0, 1, 1, 0.0])
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

    if is_scared > 0:
        layers = set_ghost_colours_scared(layers)
    else:
        layers = set_ghost_colours(layers)

    layer_1, layer_2, layer_3 = layers

    layer_1 = layer_1.at[0, 0].set(0)
    layer_2 = layer_2.at[0, 0].set(0)
    layer_3 = layer_3.at[0, 0].set(0.6)

    obs = [layer_1, layer_2, layer_3]
    rgb = jnp.stack(obs, axis=-1)

    expand_rgb = jax.numpy.kron(rgb, jnp.ones((n, n, 1)))
    layer_1 = expand_rgb[:, :, 0]
    layer_2 = expand_rgb[:, :, 1]
    layer_3 = expand_rgb[:, :, 2]

    # place normal pellets
    for i in range(len(idx)):
        if jnp.array(idx[i]).sum != 0:
            loc = idx[i]
            c = loc[1] * n + 1
            r = loc[0] * n + 1
            layer_1 = layer_1.at[c, r].set(1.0)
            layer_2 = layer_2.at[c, r].set(0.8)
            layer_3 = layer_3.at[c, r].set(0.6)

    layers = (layer_1, layer_2, layer_3)

    # Draw details
    def set_ghost_colours_details(
        layers: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        layer_1, layer_2, layer_3 = layers
        for i in range(4):
            y = ghost_pos[i][0]
            x = ghost_pos[i][1]
            c = x * n + 1
            r = y * n + 1

            layer_1 = layer_1.at[c, r].set(cr[i])
            layer_2 = layer_2.at[c, r].set(cg[i])
            layer_3 = layer_3.at[c, r].set(cb[i])

            # Make notch in top
            layer_1 = layer_1.at[c - 1, r - 1].set(0.0)
            layer_2 = layer_2.at[c - 1, r - 1].set(0.0)
            layer_3 = layer_3.at[c - 1, r - 1].set(0.0)

            # Make notch in top
            layer_1 = layer_1.at[c - 1, r + 1].set(0.0)
            layer_2 = layer_2.at[c - 1, r + 1].set(0.0)
            layer_3 = layer_3.at[c - 1, r + 1].set(0.0)

            # Eyes
            layer_1 = layer_1.at[c, r + 1].set(1)
            layer_2 = layer_2.at[c, r + 1].set(1)
            layer_3 = layer_3.at[c, r + 1].set(1)

            layer_1 = layer_1.at[c, r - 1].set(1)
            layer_2 = layer_2.at[c, r - 1].set(1)
            layer_3 = layer_3.at[c, r - 1].set(1)

        return layer_1, layer_2, layer_3

    def set_ghost_colours_scared_details(
        layers: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        layer_1, layer_2, layer_3 = layers
        for i in range(4):
            y = ghost_pos[i][0]
            x = ghost_pos[i][1]

            c = x * n + 1
            r = y * n + 1

            layer_1 = layer_1.at[x * n + 1, y * n + 1].set(0)
            layer_2 = layer_2.at[x * n + 1, y * n + 1].set(0)
            layer_3 = layer_3.at[x * n + 1, y * n + 1].set(1)

            # Make notch in top
            layer_1 = layer_1.at[c - 1, r - 1].set(0.0)
            layer_2 = layer_2.at[c - 1, r - 1].set(0.0)
            layer_3 = layer_3.at[c - 1, r - 1].set(0.0)

            # Make notch in top
            layer_1 = layer_1.at[c - 1, r + 1].set(0.0)
            layer_2 = layer_2.at[c - 1, r + 1].set(0.0)
            layer_3 = layer_3.at[c - 1, r + 1].set(0.0)

            # Eyes
            layer_1 = layer_1.at[c, r + 1].set(1)
            layer_2 = layer_2.at[c, r + 1].set(0.6)
            layer_3 = layer_3.at[c, r + 1].set(0.2)

            layer_1 = layer_1.at[c, r - 1].set(1)
            layer_2 = layer_2.at[c, r - 1].set(0.6)
            layer_3 = layer_3.at[c, r - 1].set(0.2)

        return layer_1, layer_2, layer_3

    if is_scared > 0:
        layers = set_ghost_colours_scared_details(layers)
    else:
        layers = set_ghost_colours_details(layers)

    layer_1, layer_2, layer_3 = layers

    # Power pellet is pink
    for i in range(len(pellets_loc)):
        p = pellets_loc[i]
        layer_1 = layer_1.at[p[1] * n + 2, p[0] * n + 1].set(1)
        layer_2 = layer_2.at[p[1] * n + 1, p[0] * n + 1].set(0.8)
        layer_3 = layer_3.at[p[1] * n + 1, p[0] * n + 1].set(0.6)

    # Set player is yellow
    layer_1 = layer_1.at[player_loc.x * n + 1, player_loc.y * n + 1].set(1)
    layer_2 = layer_2.at[player_loc.x * n + 1, player_loc.y * n + 1].set(1)
    layer_3 = layer_3.at[player_loc.x * n + 1, player_loc.y * n + 1].set(0)

    obs = [layer_1, layer_2, layer_3]
    rgb = jnp.stack(obs, axis=-1)
    expand_rgb

    return rgb
