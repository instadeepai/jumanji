from enum import Enum

from ic_routing_board_generation.board_generator.bfs_board import BFSBoard
from ic_routing_board_generation.board_generator.bfs_board_variations import \
    BFSBoardMinBends, BFSBoardFifo, BFSBoardShortest, BFSBoardLongest
from ic_routing_board_generation.board_generator.board_generator_random_walk_rb import \
    RandomWalkBoard

from ic_routing_board_generation.board_generator.lsystem_board import \
    LSystemBoardGen


class BoardName(str, Enum):
    """Enum of implemented board generators."""
    RANDOM_WALK = "random_walk"
    BFS_BASE = "bfs_base"
    BFS_MIN_BENDS = "bfs_min_bend"
    BFS_FIFO = "bfs_fifo"
    BFS_SHORTEST = "bfs_short"
    BFS_LONGEST = "bfs_long"
    LSYSTEMS = "lsystems_standard"


class BoardGenerator:
    """Maps BoardGeneratorType to class of generator."""
    board_generator_dict = {
        BoardName.RANDOM_WALK: RandomWalkBoard,
        BoardName.BFS_BASE: BFSBoard,
        BoardName.BFS_MIN_BENDS: BFSBoardMinBends,
        BoardName.BFS_FIFO: BFSBoardFifo,
        BoardName.BFS_SHORTEST: BFSBoardShortest,
        BoardName.BFS_LONGEST: BFSBoardLongest,
        BoardName.LSYSTEMS: LSystemBoardGen,

    }

    @classmethod
    def get_board_generator(cls, board_enum: BoardName):
        """Return class of desired board generator."""
        return cls.board_generator_dict[board_enum]
