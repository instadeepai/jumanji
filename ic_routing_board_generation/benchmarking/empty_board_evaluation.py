import time

import numpy as np
from matplotlib import pyplot as plt

from ic_routing_board_generation.benchmarking.benchmark_data_model import \
    BoardGenerationParameters
from ic_routing_board_generation.board_generator.bfs_2 import BFS_Board
from ic_routing_board_generation.board_generator.ugo_generator import BFSBoard
from ic_routing_board_generation.ic_routing.instance_generator import \
    UniversalInstanceGenerator
from ic_routing_board_generation.interface.board_generator_interface import \
    BoardGenerator, BoardGenerators


class EvaluateEmptyBoard:
    def __init__(self, empty_board: np.ndarray):
        self.training_board = empty_board
        self.empty_slot_score = -1
        self.end_score = 3
        self.wire_score = 2

    def assess_board(self):
        is_zero_mask = self.training_board == 0
        divisible_by_three_mask = self.training_board % 3 == 0

        empty_slots_mask = np.array(is_zero_mask, dtype=int) * self.empty_slot_score
        heads = np.array(self.training_board % 3 == 1, dtype=int) * self.end_score
        targets = np.array(divisible_by_three_mask * ~is_zero_mask, dtype=int) * self.end_score
        routes = np.array(self.training_board % 3 == 2, dtype=int) * self.wire_score

        return np.sum([empty_slots_mask, heads, targets, routes], axis=0)

    def score_from_neighbours(self):
        individual_score = self.assess_board()
        padded_array = np.pad(individual_score, 1, mode="constant")
        filter = np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ]
        )
        scores = []
        for row in range(1, len(padded_array) - 1):
            for column in range(1, len(padded_array[0]) - 1):
                window = padded_array[row - 1: row + 2, column - 1: column + 2]
                scores.append(np.sum(window * filter))
        scored_array = np.array(scores).reshape((len(self.training_board), len(self.training_board[0])))
        return scored_array


def plot_heatmap(scores):
    fig, ax = plt.subplots()
    im = ax.imshow(scores, cmap='Purples', interpolation='nearest')
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()


def generate_n_boards(
    board_parameters: BoardGenerationParameters,
    number_of_boards: int,
):
    board_class = BoardGenerator.get_board_generator(board_parameters.generator_type)
    board_generator = board_class(
        board_parameters.rows, board_parameters.columns,
        board_parameters.number_of_wires,
    )

    sum_all_boards = np.zeros([board_parameters.rows, board_parameters.columns])
    for _ in range(number_of_boards):
        board = board_generator.return_filled_board()
        scored_board = EvaluateEmptyBoard(board).score_from_neighbours()
        sum_all_boards += scored_board

    mean_scored_board = sum_all_boards / number_of_boards
    plot_heatmap(scores=mean_scored_board)


if __name__ == '__main__':
    board_params = BoardGenerationParameters(8, 8, 5, BoardGenerators.BFS)
    generate_n_boards(board_params, 1)
