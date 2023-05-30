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

"""Random maze geneator.

The algorithm used to generate a maze is called the recursive division method:

> Begin with the maze's space with no walls. Call this a chamber. Divide the chamber with a
> randomly positioned wall (or multiple walls) where each wall contains a randomly positioned
> passage opening within it. Then recursively repeat the process on the subchambers until all
> chambers are minimum sized.

(from [wikipedia](https://en.wikipedia.org/wiki/Maze_generation_algorithm))

It is modified to be jit-table. The ending condition of recursive function in Jax cannot depend
on abstract tensor values. Hence, instead of using recursion, a `Stack` is used to keep track
of the remaining chambers to split. While the stack is not empty, pop a chamber from the stack,
split it, and push the two newly created subchambers on the stack if they are not of minimum size,
i.e. of shape 1 by 1.

Unlike the graph based maze representation where a wall between two cells corresponds to the lack of
an edge between the two associated nodes, walls have a thickness of 1 in this pixel based
representation. Because of this, vertical walls will have an odd x coordinate while horizontal walls
will have an odd y coordinate. It also means that a passage (corresponding to an edge between two
nodes) through a vertical wall must be at an even y coordinate while a passage through a horizontal
wall must be at an even x coordinate.
"""
from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.commons.maze_utils.stack import (
    Stack,
    create_stack,
    empty_stack,
    stack_pop,
    stack_push,
)

EMPTY = 0
WALL = 1


class MazeGenerationState(NamedTuple):
    """The state of the maze generation.

    - maze: the maze containing the walls created so far.
    - chambers: the stack of remaining chambers to split.
    - key: the Jax random generation key.
    """

    maze: chex.Array
    chambers: Stack
    key: chex.PRNGKey


def create_chambers_stack(maze_width: int, maze_height: int) -> Stack:
    """Initialize the stack of chambers."""
    max_num_chamber = maze_width * maze_height
    # A chamber is defined by 4 digits: x0, y0, width, height
    chambers = create_stack(max_num_chamber, 4)
    # Initially only one chamber: the whole maze
    return stack_push(chambers, jnp.array([0, 0, maze_width, maze_height]))


def create_empty_maze(width: int, height: int) -> chex.Array:
    """Create an empty maze."""
    return jnp.full((height, width), EMPTY, dtype=jnp.int8)


def random_even(key: chex.PRNGKey, max_val: int) -> chex.Array:
    """Randomly draw an even integer between 0 (inclusive) and max_val (exclusive)."""
    return jax.random.randint(key, (), 0, (max_val + 1) // 2) * 2


def random_odd(key: chex.PRNGKey, max_val: int) -> chex.Array:
    """Randomly draw an odd integer between 0 (inclusive) and max_val (exclusive)."""
    return jax.random.randint(key, (), 0, max_val // 2) * 2 + 1


def draw_horizontal_wall(maze: chex.Array, x: int, y: int, width: int) -> chex.Array:
    """Draw a horizontal wall on the maze starting from (x,y) with the specified width."""

    def body_fun(i: int, maze: chex.Array) -> chex.Array:
        return maze.at[y, i].set(WALL)

    return jax.lax.fori_loop(x, x + width, body_fun, maze)


def draw_vertical_wall(maze: chex.Array, x: int, y: int, height: int) -> chex.Array:
    """Draw a vertical wall on the maze starting from (x,y) with the specified height."""

    def body_fun(i: int, maze: chex.Array) -> chex.Array:
        return maze.at[i, x].set(WALL)

    return jax.lax.fori_loop(y, y + height, body_fun, maze)


def create_chamber(chambers: Stack, x: int, y: int, width: int, height: int) -> Stack:
    """Create a new chamber from (x,y) and a given width and height.

    If the new chamber is less than the minimum size (1 by 1), then do nothing.
    """
    new_stack: Stack = jax.lax.cond(
        (width > 1) & (height > 1),
        lambda c: stack_push(c, jnp.array([x, y, width, height])),
        lambda c: c,
        chambers,
    )
    return new_stack


def split_vertically(
    state: MazeGenerationState, chamber: chex.Array
) -> MazeGenerationState:
    """Split the chamber vertically.

    Randomly draw a horizontal wall to split the chamber vertically. Randomly open a passage
    within this wall, and push the two newly created sub-chambers to the stack if they are not
    of minimum size.
    """
    x, y, width, height = chamber
    key, wall_key, passage_key = jax.random.split(state.key, num=3)

    # Randomly draw a vertical wall to split the chamber
    wall_dy = random_odd(wall_key, height)
    wall_y = y + wall_dy
    maze = draw_horizontal_wall(state.maze, x, wall_y, width)

    # Create chambers above and below the wall
    chambers = create_chamber(state.chambers, x, y, width, wall_dy)
    chambers = create_chamber(chambers, x, wall_y + 1, width, height - wall_dy - 1)

    # Randomly position a passage opening within the wall
    passage_x = random_even(passage_key, width)
    maze = maze.at[wall_y, x + passage_x].set(EMPTY)

    return MazeGenerationState(maze, chambers, key)


def split_horizontally(
    state: MazeGenerationState, chamber: chex.Array
) -> Tuple[chex.Array, Stack, chex.PRNGKey]:
    """Split the chamber horizontally.

    Randomly draw a vertical wall to split the chamber horizontally. Randomly open a passage
    within this wall, and push the two newly created sub-chambers to the stack if they are not
    of minimum size.
    """
    x, y, width, height = chamber
    key, wall_key, passage_key = jax.random.split(state.key, num=3)

    # Randomly draw a vertical wall to split the chamber
    wall_dx = random_odd(wall_key, width)
    wall_x = x + wall_dx
    maze = draw_vertical_wall(state.maze, wall_x, y, height)

    # Create chambers left and right of the wall
    chambers = create_chamber(state.chambers, x, y, wall_dx, height)
    chambers = create_chamber(chambers, wall_x + 1, y, width - wall_dx - 1, height)

    # Randomly position a passage opening withing the wall
    passage_y = random_even(passage_key, height)
    maze = maze.at[y + passage_y, wall_x].set(EMPTY)

    return MazeGenerationState(maze, chambers, key)


def split_next_chamber(state: MazeGenerationState) -> MazeGenerationState:
    """Split the next chamber on top of the stack."""
    chambers, chamber = stack_pop(state.chambers)
    *_, width, height = chamber

    new_state: MazeGenerationState = jax.lax.cond(
        width >= height,
        split_horizontally,
        split_vertically,
        MazeGenerationState(state.maze, chambers, state.key),
        chamber,
    )
    return new_state


def chambers_remaining(state: MazeGenerationState) -> int:
    """Check if there is any chamber remaining to split."""
    return ~empty_stack(state.chambers)


def generate_maze(width: int, height: int, key: chex.PRNGKey) -> chex.Array:
    """Randomly generate a maze.

    Args:
        width: the number of columns of the maze to create.
        height: the number of rows of the maze to create.
        key: the Jax random number generation key.

    Returns:
        maze: the generated maze.
    """
    maze = create_empty_maze(width, height)
    chambers = create_chambers_stack(width, height)

    initial_state = MazeGenerationState(maze, chambers, key)

    final_state = jax.lax.while_loop(
        chambers_remaining, split_next_chamber, initial_state
    )

    return final_state.maze
