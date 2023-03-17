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

from typing import Callable, List

import chex
from jax import numpy as jnp

from jumanji.environments.logic.rubiks_cube.constants import CubeMovementAmount, Face
from jumanji.environments.logic.rubiks_cube.types import Cube, State

# Convention:
# 0 = up face
# 1 = front face
# 2 = right face
# 3 = back face
# 4 = left face
# 5 = down face
# All read in reading order when looking directly at face
# To look directly at the faces:
# UP: LEFT face on the left and BACK face pointing up
# FRONT: LEFT face on the left and UP face pointing up
# RIGHT: FRONT face on the left and UP face pointing up
# BACK: RIGHT face on the left and UP face pointing up
# LEFT: BACK face on the left and UP face pointing up
# DOWN: LEFT face on the left and FRONT face pointing up

# Turn amounts (eg clockwise) are when looking directly at the face


def make_solved_cube(cube_size: int) -> chex.Array:
    return jnp.stack(
        [face.value * jnp.ones((cube_size, cube_size), dtype=jnp.int8) for face in Face]
    )


def is_solved(cube: Cube) -> chex.Array:
    max_sticker_by_side = jnp.max(cube, axis=(-1, -2))
    min_sticker_by_side = jnp.min(cube, axis=(-1, -2))
    return jnp.array_equal(max_sticker_by_side, min_sticker_by_side)


def sparse_reward_function(state: State) -> chex.Array:
    solved = is_solved(state.cube)
    return jnp.array(solved, float)


def do_rotation(
    cube: Cube,
    face: Face,
    amount: CubeMovementAmount,
    depth: int,
    adjacent_faces: chex.Array,
    adjacent_faces_columns: chex.Array,
    adjacent_faces_rows: chex.Array,
) -> Cube:
    """Perform the rotation, given information about which pieces move.

    Args:
        cube: the unrotated cube.
        face: which face rotates when the layer is moved.
        amount: how much to rotate by.
        depth: how many layers deep into the cube should the rotation apply to.
        adjacent_faces: array of shape (4,) indicating which faces are adjacent to the rotated
            face, in the order in which a clockwise move would be performed.
        adjacent_faces_columns: array of shape (12,) indicating for each adjacent face the column
            indices of the stickers that will move on the adjacent faces.
            For example the first 4 entries are the column indices passed through (in the order in
            which a clockwise turn would be performed) on the first adjacent face, the next 4
            correspond to the second adjacent face, and so on.
        adjacent_faces_rows: as above but for the rows.

    Returns:
        cube: the rotated cube.
    """
    cube_size = cube.shape[-1]
    if depth == 0:
        cube = cube.at[face.value].set(jnp.rot90(cube[face.value], k=-amount.value))
    adjacent_faces = jnp.repeat(adjacent_faces, cube_size)
    cube = cube.at[adjacent_faces, adjacent_faces_rows, adjacent_faces_columns].set(
        jnp.roll(
            cube[adjacent_faces, adjacent_faces_rows, adjacent_faces_columns],
            shift=cube_size * amount.value,
        )
    )
    return cube


def generate_up_move(amount: CubeMovementAmount, depth: int) -> Callable[[Cube], Cube]:
    def up_move_function(cube: Cube) -> Cube:
        cube_size = cube.shape[-1]
        adjacent_faces = jnp.array(
            [Face.FRONT.value, Face.LEFT.value, Face.BACK.value, Face.RIGHT.value]
        )
        adjacent_faces_columns = jnp.concatenate(
            [
                jnp.arange(cube_size),
                jnp.arange(cube_size),
                jnp.arange(cube_size),
                jnp.arange(cube_size),
            ]
        )
        adjacent_faces_rows = jnp.concatenate(
            [
                jnp.repeat(depth, cube_size),
                jnp.repeat(depth, cube_size),
                jnp.repeat(depth, cube_size),
                jnp.repeat(depth, cube_size),
            ]
        )
        return do_rotation(
            cube=cube,
            face=Face.UP,
            amount=amount,
            depth=depth,
            adjacent_faces=adjacent_faces,
            adjacent_faces_columns=adjacent_faces_columns,
            adjacent_faces_rows=adjacent_faces_rows,
        )

    return up_move_function


def generate_front_move(
    amount: CubeMovementAmount, depth: int
) -> Callable[[Cube], Cube]:
    def front_move_function(cube: Cube) -> Cube:
        cube_size = cube.shape[-1]
        adjacent_faces = jnp.array(
            [Face.UP.value, Face.RIGHT.value, Face.DOWN.value, Face.LEFT.value]
        )
        adjacent_faces_columns = jnp.concatenate(
            [
                jnp.arange(cube_size),
                jnp.repeat(depth, cube_size),
                jnp.flip(jnp.arange(cube_size)),
                jnp.repeat(cube_size - 1 - depth, cube_size),
            ]
        )
        adjacent_faces_rows = jnp.concatenate(
            [
                jnp.repeat(cube_size - 1 - depth, cube_size),
                jnp.arange(cube_size),
                jnp.repeat(depth, cube_size),
                jnp.flip(jnp.arange(cube_size)),
            ]
        )
        return do_rotation(
            cube=cube,
            face=Face.FRONT,
            amount=amount,
            depth=depth,
            adjacent_faces=adjacent_faces,
            adjacent_faces_columns=adjacent_faces_columns,
            adjacent_faces_rows=adjacent_faces_rows,
        )

    return front_move_function


def generate_right_move(
    amount: CubeMovementAmount, depth: int
) -> Callable[[Cube], Cube]:
    def right_move_function(cube: Cube) -> Cube:
        cube_size = cube.shape[-1]
        adjacent_faces = jnp.array(
            [Face.UP.value, Face.BACK.value, Face.DOWN.value, Face.FRONT.value]
        )
        adjacent_faces_columns = jnp.concatenate(
            [
                jnp.repeat(cube_size - 1 - depth, cube_size),
                jnp.repeat(depth, cube_size),
                jnp.repeat(cube_size - 1 - depth, cube_size),
                jnp.repeat(cube_size - 1 - depth, cube_size),
            ]
        )
        adjacent_faces_rows = jnp.concatenate(
            [
                jnp.flip(jnp.arange(cube_size)),
                jnp.arange(cube_size),
                jnp.flip(jnp.arange(cube_size)),
                jnp.flip(jnp.arange(cube_size)),
            ]
        )
        return do_rotation(
            cube=cube,
            face=Face.RIGHT,
            amount=amount,
            depth=depth,
            adjacent_faces=adjacent_faces,
            adjacent_faces_columns=adjacent_faces_columns,
            adjacent_faces_rows=adjacent_faces_rows,
        )

    return right_move_function


def generate_back_move(
    amount: CubeMovementAmount, depth: int
) -> Callable[[Cube], Cube]:
    def back_move_function(cube: Cube) -> Cube:
        cube_size = cube.shape[-1]
        adjacent_faces = jnp.array(
            [Face.UP.value, Face.LEFT.value, Face.DOWN.value, Face.RIGHT.value]
        )
        adjacent_faces_columns = jnp.concatenate(
            [
                jnp.flip(jnp.arange(cube_size)),
                jnp.repeat(depth, cube_size),
                jnp.arange(cube_size),
                jnp.repeat(cube_size - 1 - depth, cube_size),
            ]
        )
        adjacent_faces_rows = jnp.concatenate(
            [
                jnp.repeat(depth, cube_size),
                jnp.arange(cube_size),
                jnp.repeat(cube_size - 1 - depth, cube_size),
                jnp.flip(jnp.arange(cube_size)),
            ]
        )
        return do_rotation(
            cube=cube,
            face=Face.BACK,
            amount=amount,
            depth=depth,
            adjacent_faces=adjacent_faces,
            adjacent_faces_columns=adjacent_faces_columns,
            adjacent_faces_rows=adjacent_faces_rows,
        )

    return back_move_function


def generate_left_move(
    amount: CubeMovementAmount, depth: int
) -> Callable[[Cube], Cube]:
    def left_move_function(cube: Cube) -> Cube:
        cube_size = cube.shape[-1]
        adjacent_faces = jnp.array(
            [Face.UP.value, Face.FRONT.value, Face.DOWN.value, Face.BACK.value]
        )
        adjacent_faces_columns = jnp.concatenate(
            [
                jnp.repeat(depth, cube_size),
                jnp.repeat(depth, cube_size),
                jnp.repeat(depth, cube_size),
                jnp.repeat(cube_size - 1 - depth, cube_size),
            ]
        )
        adjacent_faces_rows = jnp.concatenate(
            [
                jnp.arange(cube_size),
                jnp.arange(cube_size),
                jnp.arange(cube_size),
                jnp.flip(jnp.arange(cube_size)),
            ]
        )
        return do_rotation(
            cube=cube,
            face=Face.LEFT,
            amount=amount,
            depth=depth,
            adjacent_faces=adjacent_faces,
            adjacent_faces_columns=adjacent_faces_columns,
            adjacent_faces_rows=adjacent_faces_rows,
        )

    return left_move_function


def generate_down_move(
    amount: CubeMovementAmount, depth: int
) -> Callable[[Cube], Cube]:
    def down_move_function(cube: Cube) -> Cube:
        cube_size = cube.shape[-1]
        adjacent_faces = jnp.array(
            [Face.FRONT.value, Face.RIGHT.value, Face.BACK.value, Face.LEFT.value]
        )
        adjacent_faces_columns = jnp.concatenate(
            [
                jnp.arange(cube_size),
                jnp.arange(cube_size),
                jnp.arange(cube_size),
                jnp.arange(cube_size),
            ]
        )
        adjacent_faces_rows = jnp.concatenate(
            [
                jnp.repeat(cube_size - 1 - depth, cube_size),
                jnp.repeat(cube_size - 1 - depth, cube_size),
                jnp.repeat(cube_size - 1 - depth, cube_size),
                jnp.repeat(cube_size - 1 - depth, cube_size),
            ]
        )
        return do_rotation(
            cube=cube,
            face=Face.DOWN,
            amount=amount,
            depth=depth,
            adjacent_faces=adjacent_faces,
            adjacent_faces_columns=adjacent_faces_columns,
            adjacent_faces_rows=adjacent_faces_rows,
        )

    return down_move_function


def generate_all_moves(cube_size: int) -> List[Callable[[Cube], Cube]]:
    return [
        f(amount, depth)
        for f in [
            generate_up_move,
            generate_front_move,
            generate_right_move,
            generate_back_move,
            generate_left_move,
            generate_down_move,
        ]
        for depth in range(cube_size // 2)
        for amount in CubeMovementAmount
    ]
