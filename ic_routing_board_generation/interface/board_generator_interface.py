from enum import Enum

from ic_routing_board_generation.board_generator.bfs_board import BFSBoard
from ic_routing_board_generation.board_generator.board_generator_v1_1_2_rb import \
    BoardV0
from ic_routing_board_generation.board_generator.board_generator_v2_0_0_rb import \
    RandomWalkBoard


class BoardName(str, Enum):
    """Enum of implemented board generators."""
    RANDOM_V0 = "random_v0"
    RANDOM_WALK = "random_walk"
    BFS_BASE = "BFS_base"


class BoardGenerator:
    """Maps BoardGeneratorType to class of generator."""
    board_generator_dict = {
        BoardName.RANDOM_V0: BoardV0,
        BoardName.RANDOM_WALK: RandomWalkBoard,
        BoardName.BFS_BASE: BFSBoard,
    }

    @classmethod
    def get_board_generator(cls, board_enum: BoardName):
        """Return class of desired board generator."""
        return cls.board_generator_dict[board_enum]
