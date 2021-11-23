import jax.numpy as jnp
import numpy as np
import pytest

import jumanji.testing.utils as test_utils


def test_assert_tree_is_finite() -> None:
    """Validates catching assertion errors from trees not being made of finite elements."""
    with pytest.raises(AssertionError, match="Tree is not a tree of finite elements."):
        test_utils.assert_tree_is_finite((jnp.nan,))
    with pytest.raises(AssertionError, match="Tree is not a tree of finite elements."):
        test_utils.assert_tree_is_finite((jnp.inf,))
    test_utils.assert_tree_is_finite((jnp.ones(()),))


def test_assert_is_jax_array_tree() -> None:
    """Validates catching assertion errors from trees not being a Jax type."""
    with pytest.raises(AssertionError, match="Tree is not a tree of jax arrays."):
        test_utils.assert_is_jax_array_tree((np.ones(()),))
    with pytest.raises(AssertionError, match="Tree is not a tree of jax arrays."):
        test_utils.assert_is_jax_array_tree((lambda x: x,))
    test_utils.assert_is_jax_array_tree((jnp.ones(()),))
