from typing import List, Tuple, Union, Dict, Optional, Tuple
import numpy as np
from ic_routing_board_generation.board_generator.abstract_board import AbstractBoard


class BoardProcessor:
    def __init__(self, board: Union[np.ndarray, AbstractBoard]) -> None:
        """Constructor for the BoardProcessor class."""
        if isinstance(board, np.ndarray):
            self.board_layout = board
        else:
            self.board_layout = board.return_solved_board()
        # print(self.board_layout)
        self.explored = np.full(self.board_layout.shape, False)
        self.paths = None
        self.heads = None
        self.tails = None
        self.process_board()

    def process_board(self):
        # Counts the number of wires, and stores the head, tails and the path of each wire
        # Returns the number of wires, the heads, tails and paths
        self.heads, self.tails, self.paths = self.get_paths()

    def get_paths(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[List[Tuple[int, int]]]]:
        """Gleans whether a number is a wire, head or target from the board layout.
                Heads: encoded as 4, 7 , 10,... (x%3==1)
                Targets: encoded as 3, 6, 9,... (y%3==0)
                Routes: encoded as 2, 5, 8,... (z%3==2)

                Returns:
                    None
        """
        # Initialize the list of heads, tails, and paths
        heads = []
        tails = []
        paths = []

        # Loop through the board layout
        for i in range(self.board_layout.shape[0]):
            for j in range(self.board_layout.shape[1]):
                # Check if the current position is a head
                if self.board_layout[i][j] % 3 == 2:
                    # Get the path
                    head_num = self.board_layout[i][j]
                    # print(head_num)
                    path = self.get_path(i, j, head_num)
                    # Append the path to the list of paths
                    paths.append(path)
                    # Append the head and tail to the list of heads and tails
                    heads.append(path[0])
                    tails.append(path[-1])
        return heads, tails, paths

    def get_path(self, i: int, j: int, head_num: int) -> List[Tuple[int, int]]:
        # Follows a path from head to tail
        # Returns the path
        path = [(i, j)]
        # Initiate list of explored positions
        while True:
            # Get the next position
            target_found, next_pos = self.get_next_pos(i, j, head_num)

            # Update the current position
            i, j = next_pos
            # Append the current position to the path
            path.append((i, j))
            # Mark the current position as explored
            self.explored[i][j] = True

            # Check if we have reached the end of the path
            if target_found:
                break
        # Append the tail to the path

        return path

    def get_next_pos(self, i: int, j: int, head_num: int) -> Optional[Tuple[bool, Tuple[int, int]]]:
        # Gets the next position in the path
        # Returns the next position
        # Define possible movements
        row = [-1, 0, 1, 0]
        col = [0, 1, 0, -1]

        wire_num = head_num - 1
        target_num = head_num + 1

        # Loop through the possible movements
        for k in range(4):
            # Get the next position
            next_i = i + row[k]
            next_j = j + col[k]
            # Check if the next position is valid
            if self.is_valid(next_i, next_j) and not self.explored[next_i][next_j]:
                # Check if the next position is part of the same wire

                if self.board_layout[next_i][next_j] == wire_num:
                    return False, (next_i, next_j)
                # Check if the next position is the target
                elif self.board_layout[next_i][next_j] == target_num:
                    return True, (next_i, next_j)
        # Return None if no valid position is found
        # print(f'head_num: {head_num}, wire_num: {wire_num}, target_num: {target_num}')

        return None

    def is_valid(self, i: int, j: int) -> bool:
        # Checks if the position is valid
        # Returns True if the position is valid, False otherwise
        return 0 <= i < len(self.board_layout) and 0 <= j < len(self.board_layout[0])

    def get_path_length(self, path: List[Tuple[int, int]]) -> int:
        # Returns the length of a path
        return len(path)

    def count_path_bends(self, path: List[Tuple[int, int]]) -> int:
        # Counts the number of bends in a path
        # Returns the number of bends
        bends = 0
        for i in range(1, len(path) - 1):
            # Get the previous and next positions
            prev_pos = path[i - 1]
            next_pos = path[i + 1]
            # Check if the current position is a bend
            if self.is_bend(prev_pos, path[i], next_pos):
                bends += 1
        return bends

    def is_bend(self, prev_pos: Tuple[int, int], pos: Tuple[int, int], next_pos: Tuple[int, int]) -> bool:
        # Checks if a position is a bend
        # Returns True if the position is a bend, False otherwise
        # Get the row and column of the previous and next positions
        prev_row, prev_col = prev_pos
        next_row, next_col = next_pos
        # Get the row and column of the current position
        row, col = pos
        # Check if the current position is a bend
        if (row == prev_row and row == next_row) or (col == prev_col and col == next_col):
            return False
        return True

    def proportion_filled(self) -> float:
        # Returns the percentage of the board that is filled
        # Get the number of filled positions
        filled_positions = np.count_nonzero(self.board_layout)
        # Get the total number of positions
        total_positions = self.board_layout.shape[0] * self.board_layout.shape[1]
        # Return the percentage of filled positions
        return filled_positions / total_positions

    def remove_wire(self, wire_index: int) -> None:
        # Removes a wire from the board
        # Returns None
        # Get the head and tail of the wire
        head = self.heads[wire_index]
        tail = self.tails[wire_index]
        # Get the path of the wire
        path = self.paths[wire_index]
        # Remove the wire from the board
        for pos in path:
            self.board_layout[pos[0]][pos[1]] = 0
        # Remove the wire from the list of heads, tails and paths
        self.heads.pop(wire_index)
        self.tails.pop(wire_index)
        self.paths.pop(wire_index)

        assert len(self.heads) == len(self.tails) == len(self.paths), "Heads, tails and paths not of equal length"
        assert head not in self.heads, "Head not removed"
        assert tail not in self.tails, "Tail not removed"
        assert path not in self.paths, "Path not removed"

    def get_board_layout(self) -> np.ndarray:
        # Returns the board layout
        return self.board_layout

    def get_board_statistics(self) -> Dict[str, Union[int, float]]:
        # Returns, number of wires, lengths of wires, average length of wires, number of bends in each wire,
        # average number of bends in each wire
        num_wires = len(self.heads)
        wire_lengths = [self.get_path_length(path) for path in self.paths]
        avg_wire_length = sum(wire_lengths) / num_wires
        wire_bends = [self.count_path_bends(path) for path in self.paths]
        avg_wire_bends = sum(wire_bends) / num_wires
        proportion_filled = self.proportion_filled()

        # Print summary
        # print(f'Number of wires: {num_wires}')
        # print(f'Wire lengths: {wire_lengths}')
        # print(f'Average wire length: {avg_wire_length}')
        # print(f'Wire bends: {wire_bends}')
        # print(f'Average wire bends: {avg_wire_bends}')

        # Return summary dict
        summary_dict = dict(num_wires=num_wires, wire_lengths=wire_lengths, avg_wire_length=avg_wire_length,
                            wire_bends=wire_bends, avg_wire_bends=avg_wire_bends, percent_filled=proportion_filled)

        return summary_dict


if __name__ == '__main__':
    # Create a board
    board = np.array([[22, 20, 20, 15, 2, 2, 3, 0, 0, 0],
                      [13, 0, 20, 14, 2, 30, 29, 29, 29, 29],
                      [11, 0, 20, 14, 2, 2, 4, 28, 27, 29]
                         , [11, 0, 20, 14, 14, 14, 14, 9, 8, 29]
                         , [11, 0, 20, 20, 21, 0, 16, 0, 8, 29]
                         , [11, 7, 5, 5, 5, 5, 5, 5, 8, 29]
                         , [11, 11, 11, 11, 11, 12, 0, 6, 8, 29]
                         , [0, 18, 19, 0, 0, 8, 8, 8, 8, 29]
                         , [0, 0, 0, 0, 0, 10, 25, 0, 0, 29]
                         , [0, 0, 0, 0, 24, 23, 23, 0, 0, 31]])
    # Create the boardprocessor
    boardprocessor = BoardProcessor(board)

    # Get the board layout
    board_layout = boardprocessor.get_board_layout()
    # Print the board layout
    print(board_layout)

    # Get the heads, tails and paths
    # heads, tails, paths = boardprocessor.get_paths()
    # Print the heads, tails and paths
    # Get Board Statistics
    boardprocessor.get_board_statistics()
    # Print the heads, tails and paths
    print(f'Heads: {boardprocessor.heads}')
    print(f'Tails: {boardprocessor.tails}')
    print(f'Paths: {boardprocessor.paths}')

    # Remove a wire
    boardprocessor.remove_wire(0)
    # Get the heads, tails and paths
    # heads, tails, paths = boardprocessor.get_paths()
    # Get Board Statistics
    summary_dict = boardprocessor.get_board_statistics()
    # Print summary_dict
    for key, value in summary_dict.items():
        print(f'{key}: {value}')
