import re
from typing import Dict

import jax.numpy as jnp
import numpy as np
import pytest

from jumanji.testing import pytrees as pytree_test_utils


@pytest.fixture
def mixed_tree() -> Dict:
    return {
        "jnp_array": jnp.ones(3),
        "np_array": np.ones(4),
        "None": None,
        "float": 1.0,
    }


@pytest.fixture
def jax_tree() -> Dict:
    return {"leaf1": jnp.ones(10), "branch": {"leaf2": jnp.zeros(1)}}


@pytest.fixture
def np_tree() -> Dict:
    return {"leaf1": np.ones(10), "branch": {"leaf2": np.zeros(1)}}


@pytest.fixture
def jax_and_numpy_tree() -> Dict:
    return {"leaf1": jnp.ones(10), "branch": {"leaf2": np.zeros(1)}}


def test_is_equal_pytree(mixed_tree: Dict) -> None:
    """Test that `pytree_test_utils.is_equal_pytree` outputs `True` when two trees are the
    same (have the same leaves), or `False` if they are different (at least one leaf is different).
    """
    assert pytree_test_utils.is_equal_pytree(mixed_tree, mixed_tree)
    for key in ["jnp_array", "np_array", "float"]:
        mixed_tree_alt = mixed_tree.copy()
        mixed_tree_alt[key] = mixed_tree[key] + 1.0
        assert not pytree_test_utils.is_equal_pytree(mixed_tree, mixed_tree_alt)


def test_assert_trees_are_equal(mixed_tree: Dict) -> None:
    """Test that `pytree_test_utils.assert_trees_are_equal` correctly continues when two trees
    are the same (have the same leaves), or raises an AssertionError if they are different (at
    least one leaf is different).
    """
    pytree_test_utils.assert_trees_are_equal(mixed_tree, mixed_tree)
    for key in ["jnp_array", "np_array", "float"]:
        mixed_tree_alt = mixed_tree.copy()
        mixed_tree_alt[key] = mixed_tree[key] + 1.0
        with pytest.raises(
            AssertionError,
            match=re.escape("The trees differ in at least one leaf's value(s)."),
        ):
            pytree_test_utils.assert_trees_are_equal(mixed_tree, mixed_tree_alt)


def test_assert_trees_are_different(
    mixed_tree: Dict,
) -> None:
    """Test that `pytree_test_utils.assert_trees_are_different` correctly raises an AssertionError
    when two trees are the same (have the same leaves), or continues if they are different (at
    least one leaf is different).
    """
    with pytest.raises(
        AssertionError,
        match=re.escape("The trees have the same value(s) for all leaves."),
    ):
        pytree_test_utils.assert_trees_are_different(mixed_tree, mixed_tree)
    for key in ["jnp_array", "np_array", "float"]:
        mixed_tree_alt = mixed_tree.copy()
        mixed_tree_alt[key] = mixed_tree[key] + 1.0
        pytree_test_utils.assert_trees_are_different(mixed_tree, mixed_tree_alt)


def test_is_tree_with_leaves_of_type(
    jax_tree: Dict,
    np_tree: Dict,
    jax_and_numpy_tree: Dict,
) -> None:
    """
    Test that `pytree_test_utils.is_tree_with_leaves_of_type` correctly outputs `True` if a tree
    is composed exclusively of leaves of the specified type, and `False` if there is at least one
    leaf that is not of the specified type.
    """
    assert pytree_test_utils.is_tree_with_leaves_of_type(jax_tree, jnp.ndarray)
    assert pytree_test_utils.is_tree_with_leaves_of_type(np_tree, np.ndarray)
    assert not pytree_test_utils.is_tree_with_leaves_of_type(
        jax_and_numpy_tree, jnp.ndarray
    )
    assert not pytree_test_utils.is_tree_with_leaves_of_type(
        jax_and_numpy_tree, np.ndarray
    )


def test_assert_tree_with_leaves_of_type(
    jax_tree: Dict,
    np_tree: Dict,
    jax_and_numpy_tree: Dict,
) -> None:
    """
    Test that `pytree_test_utils.assert_tree_with_leaves_of_type` correctly continues if a tree
    is composed exclusively of leaves of a specified type, and raises an AssertionError if there
    is at least one leaf of a different type.
    """
    pytree_test_utils.assert_tree_with_leaves_of_type(jax_tree, jnp.ndarray)
    pytree_test_utils.assert_tree_with_leaves_of_type(np_tree, np.ndarray)
    with pytest.raises(
        AssertionError,
        match=f"The tree has at least one leaf that is not of type {jnp.ndarray}.",
    ):
        pytree_test_utils.assert_tree_with_leaves_of_type(
            jax_and_numpy_tree, jnp.ndarray
        )
    with pytest.raises(
        AssertionError,
        match=f"The tree has at least one leaf that is not of type {np.ndarray}.",
    ):
        pytree_test_utils.assert_tree_with_leaves_of_type(
            jax_and_numpy_tree, np.ndarray
        )


def test_assert_is_jax_array_tree(
    jax_tree: Dict,
    np_tree: Dict,
    jax_and_numpy_tree: Dict,
) -> None:
    """
    Test that `pytree_test_utils.assert_is_jax_array_tree` correctly continues if a tree
    is composed exclusively of leaves of jax arrays, and raises an AssertionError if there
    is at least one leaf that is not a jax array.
    """
    pytree_test_utils.assert_is_jax_array_tree(jax_tree)
    with pytest.raises(
        AssertionError,
        match=f"The tree has at least one " f"leaf that is not of type {jnp.ndarray}.",
    ):
        pytree_test_utils.assert_is_jax_array_tree(np_tree)
    with pytest.raises(
        AssertionError,
        match=f"The tree has at least one " f"leaf that is not of type {jnp.ndarray}.",
    ):
        pytree_test_utils.assert_is_jax_array_tree(jax_and_numpy_tree)
