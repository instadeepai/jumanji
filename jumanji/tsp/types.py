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
    problem: array of coordinates for all cities
    position: index of current city
    visited_mask: binary mask (0/1 <--> unvisited/visited)
    order: array of city indices denoting route (-1 --> not filled yet)
    num_visited: how many cities have been visited
    """

    problem: Array  # (problem_size, 2)
    position: jnp.int32
    visited_mask: Array  # (problem_size,)
    order: Array  # (problem_size,)
    num_visited: jnp.int32


class Observation(NamedTuple):
    """
    problem: array of coordinates for all cities
    start_position: index of starting city
    position: index of current city
    action_mask: binary mask (0/1 <--> legal/illegal)
    """

    problem: Array
    start_position: jnp.int32
    position: jnp.int32
    action_mask: Array
