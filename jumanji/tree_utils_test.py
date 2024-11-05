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

from typing import TypeVar

import chex
import jax.numpy as jnp
import pytest

from jumanji.testing.pytrees import assert_trees_are_equal
from jumanji.tree_utils import tree_add_element, tree_slice, tree_transpose

T = TypeVar("T")


def test_tree_transpose() -> None:
    """Validates the transposition of a list of trees."""
    tree_1 = {"a": 0, "b": jnp.array([1, 2], int)}
    tree_2 = {"a": 5, "b": jnp.array([3, 4], int)}
    list_of_trees = [tree_1, tree_2]
    transposed_tree: chex.ArrayTree = {
        "a": jnp.array([0, 5], int),
        "b": jnp.array([[1, 2], [3, 4]], int),
    }
    assert_trees_are_equal(transposed_tree, tree_transpose(list_of_trees))


def test_tree_slice() -> None:
    """Validate slicing a tree."""
    tree = {"a": jnp.array([0, 5], int), "b": jnp.array([1, 2, 3], int)}
    sliced_tree: chex.ArrayTree = {"a": jnp.array(5, int), "b": jnp.array(2, int)}
    assert_trees_are_equal(sliced_tree, tree_slice(tree, 1))


@pytest.mark.parametrize(
    ["tree", "i", "element", "expected_tree"],
    [
        ((jnp.array([3, 6]),), 1, (1,), (jnp.array([3, 1]),)),
        (
            {"a": jnp.array([0, 1]), "b": (jnp.array([-1, -1]),)},
            0,
            {"a": 4, "b": (2,)},
            {"a": jnp.array([4, 1]), "b": (jnp.array([2, -1]),)},
        ),
    ],
)
def test_tree_add_element(tree: T, i: chex.Numeric, element: T, expected_tree: T) -> None:
    assert_trees_are_equal(tree_add_element(tree, i, element), expected_tree)
