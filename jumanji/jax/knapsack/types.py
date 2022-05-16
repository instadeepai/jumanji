from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

import jax.numpy as jnp
from chex import Array


@dataclass
class State:
    """
    problems: array of weights/values of the items
    last_item: index of the last added item (useless, but Pomo paper does it to match TSP setting)
    first item: index of the first added item (useless, but Pomo paper does it to match TSP setting)
    used_mask: binary mask (0/1 <--> unused/used)
    num_steps: how many steps have been taken
    remaining_budget: the budget currently remaining
    """

    problem: Array  # (problem_size, 2)
    last_item: jnp.int32
    first_item: jnp.int32
    used_mask: Array  # (problem_size,)
    num_steps: jnp.int32
    remaining_budget: jnp.float32


class Observation(NamedTuple):
    """
    problem: array of weights/values of the items
    first_item: index of first added item (useless, but Pomo paper does it to match TSP environment)
    last_item: index of the last added item
    invalid_mask: binary mask (0/1 <--> legal/illegal)
    """

    problem: Array
    first_item: jnp.int32
    last_item: jnp.int32
    invalid_mask: Array
