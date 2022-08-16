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

import jax.numpy as jnp
import pytest

from jumanji.binpack.space import Space


@pytest.fixture
def space() -> Space:
    return Space(x1=0, x2=1, y1=0, y2=1, z1=0, z2=1)


def test_space__astype(space: Space) -> None:
    assert isinstance(space.x1, int)
    space_jnp_int = space.astype(jnp.int32)
    assert space_jnp_int.x1.dtype == jnp.int32
    space_jnp_float = space.astype(jnp.float32)
    assert space_jnp_float.x1.dtype == jnp.float32


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
