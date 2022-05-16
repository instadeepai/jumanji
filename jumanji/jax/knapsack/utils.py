from chex import Array, PRNGKey
from jax import numpy as jnp
from jax import random


def compute_value_items(problem: Array, used_mask: Array) -> Array:
    """
    Compute the value of the items in the knapsack.
    """
    return jnp.dot(used_mask, problem[:, 1])


def generate_problem(problem_key: PRNGKey, problem_size: jnp.int32) -> Array:
    return random.uniform(problem_key, (problem_size, 2), minval=0, maxval=1)


def generate_first_item(start_key: PRNGKey, problem_size: jnp.int32) -> jnp.int32:
    return random.randint(start_key, (), minval=0, maxval=problem_size)
