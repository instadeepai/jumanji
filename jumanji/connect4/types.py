from typing import TYPE_CHECKING

from chex import Array
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

Board: TypeAlias = Array


@dataclass
class State:
    current_player: int
    board: Board


@dataclass
class Observation:
    board: Board
    action_mask: Array
