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

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.logic.rubiks_cube.constants import CubeMovementAmount, Face
from jumanji.environments.logic.rubiks_cube.types import Cube, State
from jumanji.environments.logic.rubiks_cube.utils import scramble_solved_cube


class Generator(abc.ABC):
    """Base class for generators for the `RubiksCube` environment."""

    def __init__(self, cube_size: int):
        """Initialises a `RubiksCube` generator for resetting the environment.

        Args:
            cube_size: the size of the cube to generate instances for.
        """
        if cube_size < 2:
            raise ValueError(
                f"Cannot meaningfully construct a cube smaller than 2x2x2, "
                f"but received cube_size={cube_size}"
            )
        self.cube_size = cube_size

    @abc.abstractmethod
    def generate_cube(self, key: chex.PRNGKey) -> Cube:
        """Generate a cube for this instance"""

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `RubiksCube` state.

        Returns:
            A `RubiksCube` state.
        """
        key, scramble_key = jax.random.split(key)
        cube = self.generate_cube(key=scramble_key)
        step_count = jnp.array(0, jnp.int32)
        return State(
            cube=cube,
            step_count=step_count,
            key=key,
        )


class ScramblingGenerator(Generator):
    """Generates instances by applying a given number of scrambles to a solved cube."""

    def __init__(
        self,
        cube_size: int,
        num_scrambles_on_reset: int,
    ):
        """
        Args:
            num_scrambles_on_reset: the number of scrambles done from a solved Rubik's Cube in the
            generation of a random instance. The lower, the closer to a solved cube the reset
            state is.
        """
        if num_scrambles_on_reset < 0:
            raise ValueError(
                f"The num_scrambles_on_reset must be non-negative, "
                f"but received num_scrambles_on_reset={num_scrambles_on_reset}"
            )
        self.num_scrambles_on_reset = num_scrambles_on_reset
        super().__init__(cube_size=cube_size)

    def generate_actions_for_scramble(self, key: chex.PRNGKey) -> chex.Array:
        return jax.random.randint(
            key=key,
            minval=0,
            maxval=len(Face) * (self.cube_size // 2) * len(CubeMovementAmount),
            shape=(self.num_scrambles_on_reset,),
            dtype=jnp.int32,
        )

    def generate_cube(self, key: chex.PRNGKey) -> Cube:
        flattened_actions_in_scramble = self.generate_actions_for_scramble(key=key)
        return scramble_solved_cube(
            flattened_actions_in_scramble=flattened_actions_in_scramble,
            cube_size=self.cube_size,
        )
