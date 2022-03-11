from enum import IntEnum
from typing import TYPE_CHECKING, NamedTuple, Union

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

import jax.numpy as jnp
from chex import Array, PRNGKey


class Position(NamedTuple):
    row: Array
    col: Array

    def __eq__(self, other: object) -> Union[bool, Array]:
        if not isinstance(other, Position):
            return NotImplemented
        return jnp.logical_and(
            jnp.all(self.row == other.row), jnp.all(self.col == other.col)
        )


@dataclass
class State:
    key: PRNGKey
    body_state: jnp.ndarray
    head_pos: Position
    fruit_pos: Position
    length: int
    step: int


class Actions(IntEnum):
    UP: int = 0
    RIGHT: int = 1
    DOWN: int = 2
    LEFT: int = 3
