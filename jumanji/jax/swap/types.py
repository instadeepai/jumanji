from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

from chex import Array, PRNGKey


@dataclass
class State:
    key: PRNGKey
    agent_pos: Array
    blue_pos: Array
    red_pos: Array
    step_count: int


class Actions(IntEnum):
    UP: int = 0
    RIGHT: int = 1
    DOWN: int = 2
    LEFT: int = 3
