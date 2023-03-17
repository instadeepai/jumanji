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

from typing import Any, Type, TypeVar

import chex
import jax
import jax.tree_util
import numpy as np
import tree as tree_lib

# In these testing utils we often operate on trees composed of leaves that we can call
# `np.asarray(leaf)` on. This allows for backend agnostic functionality (e.g. will work for tf
# tensors or jax arrays, as well as floats, strings, None etc). We define the below type to
# indicate this behavior.

MixedTypeTree = TypeVar("MixedTypeTree")


def is_equal_pytree(tree1: MixedTypeTree, tree2: MixedTypeTree) -> bool:
    """Returns true if all leaves in `tree1` and `tree2` are equal. Requires that `tree1` and
    `tree2` share the same structure, and that `np.asarray(leaf)` is valid for all leaves of the
    trees.

    Note that this function will block gradients between the input and output, and is
    created for use in the context of testing rather than for direct use inside RL algorithms."""
    is_equal_func = lambda leaf1, leaf2: np.array_equal(
        np.asarray(leaf1), np.asarray(leaf2)
    )
    is_equal_leaves = tree_lib.flatten(
        tree_lib.map_structure(is_equal_func, tree1, tree2)
    )
    is_equal = np.all(is_equal_leaves)
    return bool(is_equal)


def assert_trees_are_different(tree1: MixedTypeTree, tree2: MixedTypeTree) -> None:
    """Checks whether `tree1` and `tree2` have at least one leaf where they differ. Requires that
    `tree1` and `tree2` share the same structure, and that `np.asarray(leaf)` is valid for all
    leaves of the trees.

    This is useful for basic sanity checks, for example checking whether parameters are being
    updated."""
    assert not is_equal_pytree(
        tree1, tree2
    ), "The trees have the same value(s) for all leaves."


def assert_trees_are_equal(tree1: MixedTypeTree, tree2: MixedTypeTree) -> None:
    """Checks if all leaves in a `tree1` and `tree2` are equal. Requires that `tree1` and
    `tree2` share the same structure, and that `np.asarray(leaf)` is valid for all leaves of the
    trees.

    This is useful for basic sanity checks, for example checking if a checkpoint correctly
    restores a Learner's state."""
    assert is_equal_pytree(
        tree1, tree2
    ), "The trees differ in at least one leaf's value(s)."


def is_tree_with_leaves_of_type(input_tree: Any, *leaf_type: Type) -> bool:
    """Returns true if all leaves in the `input_tree` are of the specified `leaf_type`."""
    leaf_is_type_func = lambda leaf: isinstance(leaf, leaf_type)
    is_type_leaves = tree_lib.flatten(
        tree_lib.map_structure(leaf_is_type_func, input_tree)
    )
    tree_leaves_are_all_of_type = np.all(is_type_leaves)
    return bool(tree_leaves_are_all_of_type)


def assert_tree_with_leaves_of_type(input_tree: Any, *leaf_type: Type) -> None:
    """Asserts that all leaves in the `input_tree` are of the specified `leaf_type`."""
    assert is_tree_with_leaves_of_type(
        input_tree, *leaf_type
    ), "The tree has at least one leaf that is not of type {}.".format(
        " or ".join([str(type_) for type_ in leaf_type])
    )


def assert_is_jax_array_tree(input_tree: chex.ArrayTree) -> None:
    """Asserts that the `input_tree` has leaves that are exclusively of type `chex.Array`."""
    assert_tree_with_leaves_of_type(input_tree, chex.Array, type(None))


def has_at_least_rank(input_tree: chex.ArrayTree, rank: int) -> bool:
    """Indicate if all leaves have a rank greater or equal to `rank`."""
    has_rank_greater, _ = jax.tree_util.tree_flatten(
        jax.tree_util.tree_map(lambda x: x.ndim >= rank, input_tree)
    )
    has_at_least_rank_result = np.all(has_rank_greater)
    return bool(has_at_least_rank_result)
