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
