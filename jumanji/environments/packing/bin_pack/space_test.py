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
import jax.numpy as jnp
import pytest

from jumanji.environments.packing.bin_pack.space import Space


def test_space__astype(space: Space) -> None:
    assert space.x1.dtype == jnp.int32
    space_jnp_int8 = space.astype(jnp.int8)
    assert space_jnp_int8.x1.dtype == jnp.int8
    space_jnp_float32 = space.astype(jnp.float32)
    assert space_jnp_float32.x1.dtype == jnp.float32


def test_space__get_axis_value(space: Space) -> None:
    assert space.get_axis_value("x", 1) is space.x1
    assert space.get_axis_value("x", 2) is space.x2
    assert space.get_axis_value("y", 1) is space.y1
    assert space.get_axis_value("y", 2) is space.y2
    assert space.get_axis_value("z", 1) is space.z1
    assert space.get_axis_value("z", 2) is space.z2


def test_space__set_axis_value(space: Space) -> None:
    """Test that one can set all attributes (all axes and indices) of a space using dynamic
    indexing.
    """
    new_val = 3
    assert space.x1 is not new_val
    space.set_axis_value("x", 1, new_val)
    assert space.x1 is new_val

    assert space.x2 is not new_val
    space.set_axis_value("x", 2, new_val)
    assert space.x2 is new_val

    assert space.y1 is not new_val
    space.set_axis_value("y", 1, new_val)
    assert space.y1 is new_val

    assert space.y2 is not new_val
    space.set_axis_value("y", 2, new_val)
    assert space.y2 is new_val

    assert space.z1 is not new_val
    space.set_axis_value("z", 1, new_val)
    assert space.z1 is new_val

    assert space.z2 is not new_val
    space.set_axis_value("z", 2, new_val)
    assert space.z2 is new_val


def test_space__volume(space: Space) -> None:
    volume = space.volume()
    chex.assert_type(volume, float)
    assert volume == 1.0


@pytest.mark.parametrize(
    ["space1", "space2", "expected_intersection"],
    [
        (
            Space(x1=1, x2=2, y1=3, y2=4, z1=5, z2=6),
            Space(x1=1, x2=2, y1=3, y2=4, z1=5, z2=6),
            Space(x1=1, x2=2, y1=3, y2=4, z1=5, z2=6),
        ),
        (
            Space(x1=1, x2=3, y1=4, y2=7, z1=5, z2=9),
            Space(x1=0, x2=3, y1=2, y2=5, z1=6, z2=7),
            Space(x1=1, x2=3, y1=4, y2=5, z1=6, z2=7),
        ),
    ],
)
def test_space__intersection(space1: Space, space2: Space, expected_intersection: Space) -> None:
    space = space1.intersection(space2)
    assert space == expected_intersection


@pytest.mark.parametrize(
    ["space", "is_empty"],
    [
        (Space(x1=1, x2=2, y1=3, y2=4, z1=5, z2=6), False),
        (Space(x1=1, x2=1, y1=3, y2=4, z1=5, z2=6), True),
        (Space(x1=1, x2=2, y1=3, y2=3, z1=5, z2=6), True),
        (Space(x1=1, x2=2, y1=3, y2=4, z1=5, z2=5), True),
        (Space(x1=1, x2=-1, y1=3, y2=4, z1=5, z2=6), True),
    ],
)
def test_space__is_empty(space: Space, is_empty: bool) -> None:
    assert space.is_empty() == is_empty


@pytest.mark.parametrize(
    ["space1", "space2", "expected_intersect"],
    [
        (
            Space(x1=1, x2=2, y1=3, y2=4, z1=5, z2=6),
            Space(x1=1, x2=2, y1=3, y2=4, z1=5, z2=6),
            True,
        ),
        (
            Space(x1=1, x2=2, y1=3, y2=4, z1=5, z2=6),
            Space(x1=0, x2=1, y1=0, y2=1, z1=0, z2=1),
            False,
        ),
        (
            Space(x1=0, x2=1, y1=0, y2=1, z1=0, z2=1),
            Space(x1=0, x2=10, y1=0, y2=10, z1=0, z2=10),
            True,
        ),
    ],
)
def test_space__intersect(space1: Space, space2: Space, expected_intersect: bool) -> None:
    assert space1.intersect(space2) == expected_intersect


@pytest.mark.parametrize(
    ["space1", "space2", "is_included"],
    [
        (
            Space(x1=1, x2=2, y1=3, y2=4, z1=5, z2=6),
            Space(x1=1, x2=2, y1=3, y2=4, z1=5, z2=6),
            True,
        ),
        (
            Space(x1=1, x2=2, y1=3, y2=4, z1=5, z2=6),
            Space(x1=0, x2=1, y1=3, y2=4, z1=5, z2=6),
            False,
        ),
        (
            Space(x1=0, x2=1, y1=0, y2=1, z1=0, z2=1),
            Space(x1=-10, x2=10, y1=-10, y2=10, z1=-10, z2=10),
            True,
        ),
    ],
)
def test_space__is_included(space1: Space, space2: Space, is_included: bool) -> None:
    assert space1.is_included(space2) == is_included


@pytest.mark.parametrize(
    [
        "space",
        "lower_x_hyperplane",
        "upper_x_hyperplane",
        "lower_y_hyperplane",
        "upper_y_hyperplane",
        "lower_z_hyperplane",
        "upper_z_hyperplane",
    ],
    [
        (
            Space(x1=0, x2=1, y1=0, y2=1, z1=0, z2=1),
            Space(x1=-jnp.inf, x2=0, y1=-jnp.inf, y2=jnp.inf, z1=-jnp.inf, z2=jnp.inf),
            Space(x1=1, x2=jnp.inf, y1=-jnp.inf, y2=jnp.inf, z1=-jnp.inf, z2=jnp.inf),
            Space(x1=-jnp.inf, x2=jnp.inf, y1=-jnp.inf, y2=0, z1=-jnp.inf, z2=jnp.inf),
            Space(x1=-jnp.inf, x2=jnp.inf, y1=1, y2=jnp.inf, z1=-jnp.inf, z2=jnp.inf),
            Space(x1=-jnp.inf, x2=jnp.inf, y1=-jnp.inf, y2=jnp.inf, z1=-jnp.inf, z2=0),
            Space(x1=-jnp.inf, x2=jnp.inf, y1=-jnp.inf, y2=jnp.inf, z1=1, z2=jnp.inf),
        ),
    ],
)
def test_space__hyperplane(
    space: Space,
    lower_x_hyperplane: Space,
    upper_x_hyperplane: Space,
    lower_y_hyperplane: Space,
    upper_y_hyperplane: Space,
    lower_z_hyperplane: Space,
    upper_z_hyperplane: Space,
) -> None:
    assert space.hyperplane("x", "lower") == lower_x_hyperplane
    assert space.hyperplane("x", "upper") == upper_x_hyperplane
    assert space.hyperplane("y", "lower") == lower_y_hyperplane
    assert space.hyperplane("y", "upper") == upper_y_hyperplane
    assert space.hyperplane("z", "lower") == lower_z_hyperplane
    assert space.hyperplane("z", "upper") == upper_z_hyperplane
