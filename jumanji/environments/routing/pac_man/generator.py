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
import abc
from typing import Any, List

import chex
import jax.numpy as jnp

from jumanji.environments.routing.pac_man.types import Position, State


# flake8: noqa: C901
def generate_maze_from_ascii(maze: List) -> Any:
    """Generates a numpy maze from ascii"""
    ascii_maze = maze
    numpy_maze = []
    cookie_spaces = []
    powerup_spaces = []
    reachable_spaces = []
    ghost_spawns = []
    init_targets = []
    scatter_targets = []
    player_coords = None

    for x, row in enumerate(ascii_maze):
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


class Generator(abc.ABC):
    def __init__(self, maze: List):
        """Interface for pacman generator.

        Args:
            maze: ascii repsesentation of maze to create.
        """
        self.x_size = jnp.array(0, jnp.int32)
        self.y_size = jnp.array(0, jnp.int32)
        self.pellet_spaces = jnp.array([0, 0])

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> chex.Array:
        """Generate a problem instance.

        Args:
            key: random key.

        Returns:
            state: the generated state.
        """


class AsciiGenerator(Generator):
    """Generate maze from an ascii diagram."""

    def __init__(self, maze: List) -> None:
        """Instantiates an `AsciiGenerator `.

        This method takes in an ascii maze where each entry is one row of the maze.
        This maze is used to generate the initial state and has specific values to
        represent pellets, power_ups, the ghosts, the player and walls. It is
        important to note that this generator is deterministic and will always
        generate the same maze for for the same Ascii diagram.

        Args:
            maze: ascii repsesentation of maze to create.

        Returns:
            state: the generated state.
        """
        self.maze = maze
        self.map_data = generate_maze_from_ascii(self.maze)
        self.numpy_maze = jnp.array(self.map_data[0])

        self.pellet_spaces = jnp.array(self.map_data[1])
        self.powerup_spaces = jnp.array(self.map_data[2])
        self.reachable_spaces = self.map_data[3]

        self.ghost_spawns = jnp.array(self.map_data[4])
        self.player_coords = Position(y=self.map_data[5][0], x=self.map_data[5][1])
        self.init_targets = self.map_data[6]
        self.scatter_targets = jnp.array(self.map_data[7])
        self.x_size = self.numpy_maze.shape[0]
        self.y_size = self.numpy_maze.shape[1]

    def __call__(self, key: chex.PRNGKey) -> State:

        grid = self.numpy_maze
        pellets = self.pellet_spaces.shape[0]
        frightened_state_time = jnp.array(0, jnp.int32)
        pellet_locations = self.pellet_spaces
        power_up_locations = self.powerup_spaces
        player_locations = self.player_coords
        ghost_locations = self.ghost_spawns
        last_direction = jnp.array(0, jnp.int32)
        ghost_init_steps = jnp.array([0, 0, 0, 0])
        ghost_init_targets = self.init_targets
        ghost_actions = jnp.array([1, 1, 1, 1])
        old_ghost_locations = ghost_locations

        # Build the state.
        return State(
            key=key,
            grid=grid,
            pellets=jnp.array(pellets, jnp.int32),
            frightened_state_time=frightened_state_time,
            pellet_locations=pellet_locations,
            power_up_locations=power_up_locations,
            player_locations=player_locations,
            ghost_locations=ghost_locations,
            old_ghost_locations=old_ghost_locations,
            initial_player_locations=player_locations,
            initial_ghost_positions=ghost_locations,
            last_direction=last_direction,
            dead=False,
            ghost_init_steps=ghost_init_steps,
            ghost_init_targets=ghost_init_targets,
            ghost_actions=ghost_actions,
            visited_index=player_locations,
            ghost_starts=jnp.array([1, 5, 10, 15]),
            scatter_targets=self.scatter_targets,
            step_count=jnp.array(0, jnp.int32),
            ghost_eaten=jnp.array([True, True, True, True]),
            score=jnp.array(0, jnp.int32),
        )
