import jax
import jax.numpy as jnp
import numpy as np
from chex import ArrayTree


def assert_is_jax_array_tree(tree_of_arrays: ArrayTree) -> None:
    """Checks that the `tree_of_arrays`is a tree of jax arrays."""
    is_array, _ = jax.tree_flatten(
        jax.tree_map(lambda x: isinstance(x, jnp.ndarray), tree_of_arrays)
    )
    assert np.all(is_array), "Tree is not a tree of jax arrays."


def assert_tree_is_finite(tree_of_arrays: ArrayTree) -> None:
    """Checks that the `tree_of_arrays`is a tree of finite values (no NaN or inf)."""
    nodes, _ = jax.tree_flatten(jax.tree_map(lambda x: jnp.isfinite(x), tree_of_arrays))
    assert np.all(
        [node.all() for node in nodes]
    ), "Tree is not a tree of finite elements."
