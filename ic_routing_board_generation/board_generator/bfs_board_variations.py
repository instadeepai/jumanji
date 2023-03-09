import numpy as np

from ic_routing_board_generation.board_generator.bfs_board import BFSBoard


class BFSBoardMinBends(BFSBoard):
    def return_training_board(self) -> np.ndarray:
        self.reset_board()
        self.fill_board_with_clipping(2, 'min_bends', verbose=False)
        if self.filled:
            return self.empty_board
        else:
            return self.unsolved_board

    def return_solved_board(self) -> np.ndarray:
        self.reset_board()
        self.fill_board_with_clipping(int(0.5 * self.num_agents), 'min_bends', verbose=False)
        if self.filled:
            return self.empty_board
        else:
            return self.solved_board

class BFSBoardFifo(BFSBoard):
    def return_training_board(self) -> np.ndarray:
        self.reset_board()
        self.fill_board_with_clipping(int(0.5 * self.num_agents), 'fifo', verbose=False)
        if self.filled:
            return self.empty_board
        else:
            return self.unsolved_board

    def return_solved_board(self) -> np.ndarray:
        self.reset_board()
        self.fill_board_with_clipping(int(0.5 * self.num_agents), 'fifo', verbose=False)
        if self.filled:
            return self.empty_board
        else:
            return self.solved_board

class BFSBoardShortest(BFSBoard):
    def return_training_board(self) -> np.ndarray:
        self.reset_board()
        self.fill_board_with_clipping(int(0.5 * self.num_agents), 'shortest', verbose=False)
        if self.filled:
            return self.empty_board
        else:
            return self.unsolved_board

    def return_solved_board(self) -> np.ndarray:
        self.reset_board()
        self.fill_board_with_clipping(int(0.5 * self.num_agents), 'shortest', verbose=False)
        if self.filled:
            return self.empty_board
        else:
            return self.solved_board

class BFSBoardLongest(BFSBoard):
    def return_training_board(self) -> np.ndarray:
        self.reset_board()
        self.fill_board_with_clipping(int(0.5 * self.num_agents), 'longest', verbose=False)
        if self.filled:
            return self.empty_board
        else:
            return self.unsolved_board

    def return_solved_board(self) -> np.ndarray:
        self.reset_board()
        self.fill_board_with_clipping(int(0.5 * self.num_agents), 'longest', verbose=False)
        if self.filled:
            return self.empty_board
        else:
            return self.solved_board
