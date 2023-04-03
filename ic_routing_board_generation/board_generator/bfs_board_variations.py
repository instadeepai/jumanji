import numpy as np

from ic_routing_board_generation.board_generator.bfs_board import BFSBoard


class BFSBoardMinBends(BFSBoard):
    def return_training_board(self) -> np.ndarray:
        self.reset_board()
        self.fill_board_with_clipping(2, 'min_bends', verbose=True)
        if self.filled:
            return self.empty_board
        else:
            return self.unsolved_board

class BFSBoardFifo(BFSBoard):
    def return_training_board(self) -> np.ndarray:
        self.reset_board()
        self.fill_board_with_clipping(int(0.5 * self.num_agents), 'fifo', verbose=True)
        if self.filled:
            return self.empty_board
        else:
            return self.unsolved_board

class BFSBoardShortest(BFSBoard):
    def return_training_board(self) -> np.ndarray:
        self.reset_board()
        self.fill_board_with_clipping(int(0.5 * self.num_agents), 'shortest', verbose=True)
        if self.filled:
            return self.empty_board
        else:
            return self.unsolved_board

class BFSBoardLongest(BFSBoard):
    def return_training_board(self) -> np.ndarray:
        self.reset_board()
        self.fill_board_with_clipping(int(0.5 * self.num_agents), 'longest', verbose=True)
        if self.filled:
            return self.empty_board
        else:
            return self.unsolved_board
