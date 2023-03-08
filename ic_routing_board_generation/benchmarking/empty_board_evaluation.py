import numpy as np
from matplotlib import pyplot as plt

from ic_routing_board_generation.benchmarking.benchmark_data_model import \
    BoardGenerationParameters
from ic_routing_board_generation.interface.board_generator_interface import \
    BoardGenerator, BoardName


class EvaluateEmptyBoard:
    def __init__(self, empty_board: np.ndarray):
        self.training_board = empty_board
        self.empty_slot_score = -2
        self.end_score = 3
        self.wire_score = 2
        self.max_score = 14 * 4
        self.min_score = -6

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
        padded_starting_board = np.pad(self.training_board, 1, mode="constant")
        padded_array = np.pad(individual_score, 1, mode="constant")
        filter = np.array(
            [
                [0, 1, 0],
                [1, 2, 1],
                [0, 1, 0],
            ]
        )
        scores = []
        for row in range(1, len(padded_array) - 1):
            for column in range(1, len(padded_array[0]) - 1):
                training_board_filtered = padded_starting_board[row - 1: row + 2, column - 1: column + 2]  * filter
                print(training_board_filtered)
                diversity = len(np.unique(training_board_filtered / 3))
                print(diversity)

                window = padded_array[row - 1: row + 2, column - 1: column + 2]
                score = np.sum(window * filter * diversity)
                normalized_score = (score-self.min_score)/(self.max_score - self.min_score)
                scores.append(normalized_score)
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
    # board_params = BoardGenerationParameters(8, 8, 8, BoardName.BFS)
    # generate_n_boards(board_params, 1)
 #    board = np.array([[0, 0 ,0 ,7 ,5 ,5 ,5 ,5],
 # [ 0  ,0  ,0  ,0  ,4  ,0  ,0  ,5],
 # [ 0 ,10 , 8 , 8 , 2 , 0 , 0  ,5],
 # [ 0 , 0, 16 , 8 , 2 , 0 ,13  ,5],
 # [ 0 , 0 ,15 , 8 , 2 , 0 ,11 , 6],
 # [ 0 , 0 , 0 , 8 , 2 , 0 ,11  ,0],
 # [ 0 , 0 , 0 , 9 , 2 , 0 ,11 , 0],
 # [ 0 , 0 , 3 , 2 , 2, 12 ,11  ,0]])

    board = np.array([[12, 11, 11 ,11 ,11 ,11, 11, 11],
 [ 8 , 8 , 8 , 8 , 8  ,9 , 0 ,11],
 [ 8 , 2 , 2 , 2 , 2 , 2 , 2, 11],
 [ 8 , 2, 15 ,14 , 0 , 0 , 2 ,11],
 [ 8 , 2 , 2 ,14  ,0 , 0 , 4 ,13],
 [ 8,  0 , 3 ,14 , 6  ,0 , 0 , 0],
 [ 8 , 0 , 0, 14 , 5  ,0 , 0 , 0],
 [10 , 0 , 0 ,16  ,7  ,0  ,0 , 0],]


    )
    # board = np.array(
    #     [
    #         []
    #     ]
    # )
    scores = EvaluateEmptyBoard(board).score_from_neighbours()
    plot_heatmap(scores)
