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
import pytest
import scipy

from jumanji.environments.commons.maze_utils.maze_generation import (
    EMPTY,
    WALL,
    MazeGenerationState,
    create_chambers_stack,
    create_empty_maze,
    generate_maze,
    random_even,
    random_odd,
    split_horizontally,
    split_vertically,
)
from jumanji.environments.commons.maze_utils.stack import Stack, stack_pop


def no_more_chamber(maze: chex.Array) -> chex.Array:
    """Test if there is no chamber in the maze that can be divided anymore.

    A chamber can be divided if its width and height are greater or equal to two.
    To efficiently detect chambers of shape at least two by two in the maze, a convolution with
    a positive kernel of size (2,2) can be used: if there is a chamber, its convolution with
    the kernel will be 0. The contrapositive is that if there is no 0 in the convolution of
    the maze and a positive kernel of size (2,2), then there is no chamber.
    """
    kernel = jnp.ones((2, 2))
    convolved = jax.scipy.signal.convolve2d(maze, kernel, mode="valid")
    return jnp.all(convolved)


def all_tiles_connected(maze: chex.Array) -> bool:
    """Test if all the tiles of the maze can be reached.

    The function scipy.ndimage.label can be used to count the number of connected components
    in an image. The background has to be composed of zero, and connected components are convex
    shapes of positive integers pixels.
    If the maze contain only one connected component, then all tiles can be reached from
    all others.
    """
    # replace walls by 0 and empty tiles by 1
    inverted_maze = -(maze - WALL) / WALL
    _, n_connected_components = scipy.ndimage.label(inverted_maze)
    return bool(n_connected_components == 1)


class TestMazeGeneration:
    WIDTH = 15
    HEIGHT = 15

    @pytest.fixture
    def maze(self) -> chex.Array:
        return create_empty_maze(self.WIDTH, self.HEIGHT)

    @pytest.fixture
    def chambers(self) -> Stack:
        return create_chambers_stack(self.WIDTH, self.HEIGHT)

    @pytest.fixture
    def key(self) -> chex.PRNGKey:
        return jax.random.PRNGKey(0)

    def test_create_chambers_stack(self) -> None:
        chambers = create_chambers_stack(self.WIDTH, self.HEIGHT)
        assert isinstance(chambers, Stack)
        assert chambers.data.shape == (self.WIDTH * self.HEIGHT, 4)
        # Initially only one chambers: the full maze
        assert jnp.all(chambers.data[0] == jnp.array([0, 0, self.WIDTH, self.HEIGHT]))

    def test_create_empty_maze(self) -> None:
        maze = create_empty_maze(self.WIDTH, self.HEIGHT)
        # Initially there is no wall
        assert jnp.all(maze == EMPTY)
        assert maze.shape == (self.HEIGHT, self.WIDTH)

    def test_random_even(self, key: chex.PRNGKey) -> None:
        max_val = 10
        repeat = 10
        for _ in range(repeat):
            key, subkey = jax.random.split(key)
            i = random_even(subkey, max_val)
            assert i % 2 == 0
            assert 0 <= i < max_val

    def test_random_odd(self, key: chex.PRNGKey) -> None:
        max_val = 10
        repeat = 10
        for _ in range(repeat):
            key, subkey = jax.random.split(key)
            i = random_odd(subkey, max_val)
            assert i % 2 == 1
            assert 0 <= i < max_val

    def test_split_vertically(
        self, maze: chex.Array, chambers: Stack, key: chex.PRNGKey
    ) -> None:
        """Test that a horizontal wall is drawn and that subchambers are added to stack."""
        chambers, chamber = stack_pop(chambers)
        state = MazeGenerationState(maze, chambers, key)
        maze, chambers, _ = split_vertically(state, chamber)

        # Should contain (width - 1) wall tiles: full width except one passage
        assert jnp.sum(maze == WALL) == self.WIDTH - 1
        # Only one row must contain non 0 element
        assert jnp.sum(jnp.any(maze, axis=1)) == 1

        assert chambers.insertion_index >= 1

    def test_split_horizontally(
        self, maze: chex.Array, chambers: Stack, key: chex.PRNGKey
    ) -> None:
        """Test that a vertical wall is drawn and that subchambers are added to stack."""
        chambers, chamber = stack_pop(chambers)
        state = MazeGenerationState(maze, chambers, key)
        maze, chambers, _ = split_horizontally(state, chamber)

        # Should contain (height - 1) wall tiles: full height except one passage
        assert jnp.sum(maze == WALL) == self.HEIGHT - 1
        # Only one row must contain non 0 element
        assert jnp.sum(jnp.any(maze, axis=0)) == 1

        assert chambers.insertion_index >= 1

    def test_generate_maze(self, key: chex.PRNGKey) -> None:
        maze = generate_maze(self.WIDTH, self.HEIGHT, key)

        # If the maze cannot be divided anymore and all tiles are connected, it is a valid maze
        assert no_more_chamber(maze)
        assert all_tiles_connected(maze)

    def test_generate_maze_seed(self, key: chex.PRNGKey) -> None:
        key1, key2 = jax.random.split(key)

        maze1 = generate_maze(self.WIDTH, self.HEIGHT, key1)
        maze2 = generate_maze(self.WIDTH, self.HEIGHT, key2)

        assert jnp.any(maze1 != maze2)

        maze3 = generate_maze(self.WIDTH, self.HEIGHT, key1)

        assert jnp.all(maze1 == maze3)
