from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

import jax.numpy as jnp
from chex import Array, PRNGKey


@dataclass
class State:
    key: PRNGKey
    grid: Array
    step: jnp.int_
    finished_agents: Array


class Position(NamedTuple):
    x: jnp.int_
    y: jnp.int_
