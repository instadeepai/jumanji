from enum import Enum

from ic_routing_board_generation.board_generator.board_generator_v2_0_0_rb import \
    Board
from ic_routing_board_generation.board_generator.board_generator_v1_1_2_rb import Board as BoardV1
from ic_routing_board_generation.board_generator.dummy_boar_generator import \
    DummyBoard


class BoardGenerators(str, Enum):
    """Enum of implemented board generators."""
    BASELINE = "baseline random"
    RANDY_V1 = "randy_v1"
    RANDOM_ROUTE = "random route"
    BFS = "BFS"
    DUMMY = "dummy"


class BoardGenerator:
    """Maps BoardGeneratorType to class of generator."""
    board_generator_dict = {
        BoardGenerators.RANDY_V1: BoardV1,
        BoardGenerators.RANDOM_ROUTE: Board,
        BoardGenerators.DUMMY: DummyBoard,
    }

    @classmethod
    def get_board_generator(cls, board_enum: BoardGenerators):
        """Return class of desired board generator."""
        return cls.board_generator_dict[board_enum]
