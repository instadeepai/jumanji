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
import pytest
from jax import numpy as jnp

from jumanji.environments.logic.rubiks_cube.constants import DEFAULT_CUBE_SIZE, Face
from jumanji.environments.logic.rubiks_cube.reward_functions import SparseRewardFunction
from jumanji.environments.logic.rubiks_cube.types import Cube, FakeState
from jumanji.environments.logic.rubiks_cube.utils import (
    CubeMovementAmount,
    generate_all_moves,
    generate_back_move,
    generate_down_move,
    generate_front_move,
    generate_left_move,
    generate_right_move,
    generate_up_move,
    make_solved_cube,
)

# 3x3x3 moves, for testing purposes
up_move = generate_up_move(CubeMovementAmount.CLOCKWISE, 0)
up_move_inverse = generate_up_move(CubeMovementAmount.ANTI_CLOCKWISE, 0)
up_move_half_turn = generate_up_move(CubeMovementAmount.HALF_TURN, 0)
front_move = generate_front_move(CubeMovementAmount.CLOCKWISE, 0)
front_move_inverse = generate_front_move(CubeMovementAmount.ANTI_CLOCKWISE, 0)
front_move_half_turn = generate_front_move(CubeMovementAmount.HALF_TURN, 0)
right_move = generate_right_move(CubeMovementAmount.CLOCKWISE, 0)
right_move_inverse = generate_right_move(CubeMovementAmount.ANTI_CLOCKWISE, 0)
right_move_half_turn = generate_right_move(CubeMovementAmount.HALF_TURN, 0)
back_move = generate_back_move(CubeMovementAmount.CLOCKWISE, 0)
back_move_inverse = generate_back_move(CubeMovementAmount.ANTI_CLOCKWISE, 0)
back_move_half_turn = generate_back_move(CubeMovementAmount.HALF_TURN, 0)
left_move = generate_left_move(CubeMovementAmount.CLOCKWISE, 0)
left_move_inverse = generate_left_move(CubeMovementAmount.ANTI_CLOCKWISE, 0)
left_move_half_turn = generate_left_move(CubeMovementAmount.HALF_TURN, 0)
down_move = generate_down_move(CubeMovementAmount.CLOCKWISE, 0)
down_move_inverse = generate_down_move(CubeMovementAmount.ANTI_CLOCKWISE, 0)
down_move_half_turn = generate_down_move(CubeMovementAmount.HALF_TURN, 0)


def is_face_turn(cube_size: int) -> List[bool]:
    """Says whether each action from generate_all_moves is a face turn (ie with depth 0)"""
    per_face_result_true = [True] * len(CubeMovementAmount)
    per_face_result_false = [False] * (len(CubeMovementAmount) * ((cube_size // 2) - 1))
    per_face_result = per_face_result_true + per_face_result_false
    result = [per_face_result for _ in Face]
    return [item for r in result for item in r]


@pytest.mark.parametrize(
    "move, inverse_move",
    [
        (up_move, up_move_inverse),
        (front_move, front_move_inverse),
        (right_move, right_move_inverse),
        (back_move, back_move_inverse),
        (left_move, left_move_inverse),
        (down_move, down_move_inverse),
    ],
)
def test_inverses(
    differently_stickered_cube: chex.Array,
    move: Callable[[Cube], Cube],
    inverse_move: Callable[[Cube], Cube],
) -> None:
    """Test that applying a move followed by its inverse leads back to the original"""
    cube = move(differently_stickered_cube)
    cube = inverse_move(cube)
    assert jnp.array_equal(cube, differently_stickered_cube)


@pytest.mark.parametrize(
    "move, half_turn_move",
    [
        (up_move, up_move_half_turn),
        (front_move, front_move_half_turn),
        (right_move, right_move_half_turn),
        (back_move, back_move_half_turn),
        (left_move, left_move_half_turn),
        (down_move, down_move_half_turn),
    ],
)
def test_half_turns(
    differently_stickered_cube: chex.Array,
    move: Callable[[Cube], Cube],
    half_turn_move: Callable[[Cube], Cube],
) -> None:
    """Test that 2 applications of a move followed by its half turn leads back to the original"""
    cube = move(differently_stickered_cube)
    cube = move(cube)
    cube = half_turn_move(cube)
    assert jnp.array_equal(cube, differently_stickered_cube)


def test_solved_reward(
    solved_cube: chex.Array, differently_stickered_cube: chex.Array
) -> None:
    """Test that the cube fixtures have the expected rewards"""
    solved_state = FakeState(cube=solved_cube, step_count=jnp.int32(0))
    differently_stickered_state = FakeState(
        cube=differently_stickered_cube, step_count=jnp.int32(0)
    )
    assert jnp.equal(SparseRewardFunction()(solved_state), 1.0)
    assert jnp.equal(SparseRewardFunction()(differently_stickered_state), 0.0)


@pytest.mark.parametrize(
    "move, move_is_face_turn",
    zip(
        generate_all_moves(cube_size=DEFAULT_CUBE_SIZE),
        is_face_turn(cube_size=DEFAULT_CUBE_SIZE),
    ),
)
def test_moves_nontrivial(
    solved_cube: chex.Array,
    differently_stickered_cube: chex.Array,
    move: Callable[[Cube], Cube],
    move_is_face_turn: bool,
) -> None:
    """Test that all moves leave the cube in a non-solved state"""
    move_solved_cube = move(solved_cube)
    move_solved_state = FakeState(cube=move_solved_cube, step_count=jnp.int32(0))
    assert jnp.equal(SparseRewardFunction()(move_solved_state), 0.0)
    assert (
        jnp.not_equal(solved_cube, move_solved_cube).sum()
        == (len(Face) - 2) * solved_cube.shape[-1]
    )
    for face in Face:
        assert (
            jnp.equal(move_solved_cube, face.value).sum()
            == solved_cube.shape[-1] * solved_cube.shape[-1]
        )
    differently_stickered_cube_size = differently_stickered_cube.shape[-1]
    moved_differently_stickered_cube = move(differently_stickered_cube)
    num_face_impacted_cubies = (
        differently_stickered_cube_size * differently_stickered_cube_size
        - int(differently_stickered_cube_size % 2)
    )
    num_non_face_impacted_cubies = (len(Face) - 2) * differently_stickered_cube_size
    assert jnp.not_equal(
        differently_stickered_cube, moved_differently_stickered_cube
    ).sum() == num_non_face_impacted_cubies + (
        num_face_impacted_cubies if move_is_face_turn else 0
    )
    if differently_stickered_cube_size % 2 == 1:
        assert jnp.array_equal(
            differently_stickered_cube[
                :,
                differently_stickered_cube_size // 2,
                differently_stickered_cube_size // 2,
            ],
            moved_differently_stickered_cube[
                :,
                differently_stickered_cube_size // 2,
                differently_stickered_cube_size // 2,
            ],
        )


@pytest.mark.parametrize(
    "first_move, second_move",
    [(up_move, down_move), (right_move, left_move), (front_move, back_move)],
)
def test_commuting_moves(
    differently_stickered_cube: chex.Array,
    first_move: Callable[[Cube], Cube],
    second_move: Callable[[Cube], Cube],
) -> None:
    """Check that moves that should commute, do in fact commute
    (on a differently stickered cube)"""
    first_then_second = second_move(first_move(differently_stickered_cube))
    second_then_first = first_move(second_move(differently_stickered_cube))
    assert jnp.array_equal(first_then_second, second_then_first)


@pytest.mark.parametrize(
    "first_move, second_move",
    [
        (up_move, front_move),
        (up_move, right_move),
        (up_move, back_move),
        (up_move, left_move),
        (front_move, right_move),
        (front_move, left_move),
        (front_move, down_move),
        (right_move, back_move),
        (right_move, down_move),
        (back_move, left_move),
        (back_move, down_move),
        (left_move, down_move),
    ],
)
def test_non_commuting_moves(
    solved_cube: chex.Array,
    first_move: Callable[[Cube], Cube],
    second_move: Callable[[Cube], Cube],
) -> None:
    """Check that moves that should not commute, do not (on a solved cube)"""
    first_then_second = second_move(first_move(solved_cube))
    second_then_first = first_move(second_move(solved_cube))
    assert ~jnp.array_equal(first_then_second, second_then_first)


@pytest.mark.parametrize(
    "cube_size, indices",
    [
        (3, [2, 17, 5, 11, 8, 14]),
        (5, [5, 35, 11, 23, 17, 29]),
        (7, [2, 8, 47, 53, 11, 17, 29, 35, 20, 26, 38, 44]),
    ],
)
def test_checkerboard(cube_size: int, indices: List[int]) -> None:
    """Check that the checkerboard scramble gives the expected result"""
    cube = make_solved_cube(cube_size=cube_size)
    all_moves = generate_all_moves(cube_size=cube_size)
    for index in indices:
        cube = all_moves[index](cube)
    opposite_face = [Face.DOWN, Face.BACK, Face.LEFT, Face.FRONT, Face.RIGHT, Face.UP]
    for face in Face:
        expected_result = jnp.concatenate(
            [
                jnp.array([face.value, opposite_face[face.value].value])
                for _ in range((cube_size * cube_size) // 2)
            ]
            + [jnp.array([face.value])]
        ).reshape(cube_size, cube_size)
        assert jnp.array_equal(cube[face.value], expected_result)


def test_manual_scramble(
    solved_cube: chex.Array, expected_scramble_result: chex.Array
) -> None:
    """Testing a particular scramble manually.
    Scramble chosen to have all faces touched at least once"""
    scramble = [
        up_move,
        left_move_half_turn,
        down_move_inverse,
        up_move_half_turn,
        back_move_inverse,
        right_move,
        front_move,
        right_move_inverse,
        left_move_inverse,
        back_move_half_turn,
        front_move_inverse,
        up_move,
        down_move,
    ]
    cube = solved_cube
    for move in scramble:
        cube = move(cube)
    assert jnp.array_equal(expected_scramble_result, cube)
