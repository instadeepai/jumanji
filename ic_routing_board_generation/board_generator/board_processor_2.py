import random
import time
from dataclasses import dataclass
from typing import List, Union, Dict, Tuple, Optional
import numpy as np
from bfs_board import BFSBoard
from board_generator_random_walk_rb import RandomWalkBoard
from board_generator_wfc_oj import WFCBoard
from ic_routing_board_generation.board_generator.abstract_board import AbstractBoard
from lsystem_board import LSystemBoardGen

EMPTY, PATH, POSITION, TARGET = 0, 1, 2, 3  # Ideally should be imported from Jumanji

# Define exceptions for the board validity checks
class IncorrectBoardSizeError(Exception):
    """ Raised when a board size does not match the specified dimensions."""
    pass


class NumAgentsOutOfRangeError(Exception):
    """ Raised when self._wires_on_board is negative."""
    pass


class EncodingOutOfRangeError(Exception):
    """ Raised when one or more cells on the board have an invalid index."""
    pass


class DuplicateHeadsTailsError(Exception):
    """ Raised when one of the heads or tails of a wire is duplicated."""
    pass


class MissingHeadTailError(Exception):
    """ Raised when one of the heads or tails of a wire is missing."""
    pass


class InvalidWireStructureError(Exception):
    """ Raised when one or more of the wires has an invalid structure, e.g. looping or branching."""
    pass


class PathNotFoundError(Exception):
    """ Raised when a path cannot be found between a head and a target."""
    pass


class BoardProcessor:
    def __init__(self, board: Union[np.ndarray, AbstractBoard]) -> None:
        """Constructor for the BoardProcessor class."""
        if isinstance(board, np.ndarray):
            self.board_layout = board
        else:
            self.board_layout = board.return_solved_board()


        self.board = board

        self.heads, self.targets, self.paths = None, None, None
        self.rows = self.board_layout.shape[0]
        self.cols = self.board_layout.shape[1]
        # if isinstance(board, AbstractBoard):
        #     # Check that board is valid!
        #     self.is_valid_board()
        self.process_board()

        # Given a board, we want to extract the positions of the heads, targets and paths
        # We also want to check that the board is valid, i.e. that it has the correct number of wires, heads and tails

    def get_heads_and_targets(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Returns the heads and targets of the board layout
        heads are encoded as 2,5,8,11,...
        targets are encoded as 3,6,9,12,...
        both are greater than 0
        """
        # Get the heads and targets in the right order
        heads = []
        targets = []
        board_layout = self.board_layout
        # Get the maximum value in the board layout
        max_val = np.max(board_layout)
        # Get the heads and targets
        for i in range(1, max_val + 1):
            # Get the head and target
            if i % 3 == POSITION:
                try:
                    head = np.argwhere(board_layout == i)[0]
                    heads.append(tuple(head))
                    target = np.argwhere(board_layout == i + 1)[0]
                    targets.append(tuple(target))
                except IndexError:
                    print(f"IndexError: i = {i}, max_val = {max_val}, board_layout = {board_layout}")
        return heads, targets

    def get_wires_on_board(self) -> int:
        """Returns the number of wires on the board by counting the number of unique wire encodings."""
        return len(np.unique(self.board_layout[self.board_layout % 3 == PATH]))

    def process_board(self) -> None:
        """Processes the board by getting the heads, targets and paths."""

        self.heads, self.targets = self.get_heads_and_targets()
        # find the paths
        self.paths = self.get_paths_from_heads_and_targets()

    def get_paths_from_heads_and_targets(self) -> List[List[Tuple[int, int]]]:
        """Gets the paths from all heads to all targets via BFS using only valid moves and cells with wire encodings."""
        paths = []
        for i in range(len(self.heads)):
            paths.append(self.get_path_from_head_and_target(self.heads[i], self.targets[i]))
        return paths

    def get_path_from_head_and_target(self, head, target) -> List[Tuple[int, int]]:
        """Gets the path from a head to a target via BFS using only valid moves and cells with wire encodings.
        Essentially remove_extraneous_path_cells"""
        # path = [head]
        # valid moves are up, down, left and right with the bounds of the board
        valid_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # Shuffle valid moves to ensure that the search is not biased
        random.shuffle(valid_moves)
        # Get the head and target encodings
        head_encoding = self.board_layout[head]
        target_encoding = self.board_layout[target]
        path_encoding = head_encoding - 1
        # Only cells with value head_encoding, target_encoding or path_encoding are valid
        valid_cells = [head_encoding, target_encoding, path_encoding, EMPTY]
        # Initialize the queue
        queue = [head]
        # Initialize the explored array
        explored = np.full(self.board_layout.shape, False)
        # Initialize the parent array
        parent = np.full(self.board_layout.shape, None)
        # Initialize the path
        path = []

        while len(queue) > 0:
            # Get the current cell
            current_cell = queue.pop(0)
            # Mark the current cell as explored
            explored[current_cell] = True
            # Check if the current cell is the target
            if current_cell == target:
                # Get the path from the target to the head
                path = self.get_path_from_target_to_head(parent, target)
                break
            # Get the neighbours of the current cell
            neighbours = self.get_neighbours(current_cell, valid_moves, valid_cells)
            # Loop through the neighbours
            for neighbour in neighbours:
                # Check if the neighbour has been explored
                if not explored[neighbour]:
                    # Add the neighbour to the queue
                    queue.append(neighbour)
                    # Mark the neighbour as explored
                    explored[neighbour] = True
                    # Set the parent of the neighbour
                    parent[neighbour] = current_cell

        self.remove_extraneous_path_cells(path, path_encoding)

        # Raise error if path not found
        if len(path) == 0 or (head not in path) or (target not in path):
            raise PathNotFoundError
        return path

    def remove_extraneous_path_cells(self, path: List[Tuple[int, int]], path_encoding: int) -> None:
        """Removes extraneous path cells from the board layout."""
        # Change any cell with the same wire_encoding but not in the path to an empty cell
        path_set = set(path)
        for i in range(self.board_layout.shape[0]):
            for j in range(self.board_layout.shape[1]):
                if self.board_layout[i, j] == path_encoding and (i, j) not in path_set:
                    self.board_layout[i, j] = EMPTY
                elif self.board_layout[i, j] == EMPTY and (i, j) in path_set:
                    self.board_layout[i, j] = PATH

    @staticmethod
    def get_path_from_target_to_head(parent, target) -> List[Tuple[int, int]]:
        """Gets the path from a target to a head."""
        # Initialize the path
        path = [target]
        # Get the parent of the target
        parent_cell = parent[target]
        # Loop until the parent cell is None
        while parent_cell is not None:
            # Add the parent cell to the path
            path.append(parent_cell)
            # Get the parent of the parent cell
            parent_cell = parent[parent_cell]
        # Reverse the path
        path.reverse()
        return path

    def get_neighbours(self, cell, valid_moves, valid_cells) -> List[Tuple[int, int]]:
        """Gets the valid neighbours of a cell."""
        # Initialize the list of neighbours
        neighbours = []
        # Loop through the valid moves
        for move in valid_moves:
            # Get the neighbour
            neighbour = (cell[0] + move[0], cell[1] + move[1])
            # Check if the neighbour is valid
            if self.is_valid_cell(neighbour, valid_cells):
                # Add the neighbour to the list of neighbours
                neighbours.append(neighbour)
        return neighbours

    def is_valid_cell(self, cell: Tuple[int, int], valid_cells: List[Tuple[int, int]]) -> bool:
        """Checks if a cell is valid."""
        # Check if the cell is within the bounds of the board
        if cell[0] < 0 or cell[0] >= self.board_layout.shape[0] or cell[1] < 0 or cell[1] >= self.board_layout.shape[1]:
            return False
        # Check if the cell has a valid encoding
        if self.board_layout[cell] not in valid_cells:
            return False
        return True

    @staticmethod
    def get_path_length(path: List[Tuple[int, int]]) -> int:
        """Gets the length of a path"""
        return len(path)

    def count_path_bends(self, path: List[Tuple[int, int]]) -> int:
        """Counts the number of bends in a path"""
        bends = 0
        for i in range(1, len(path) - 1):
            # Get the previous and next positions
            prev_pos = path[i - 1]
            next_pos = path[i + 1]
            # Check if the current position is a bend
            if self.is_bend(prev_pos, path[i], next_pos):
                bends += 1
        return bends

    @staticmethod
    def is_bend(prev_pos: Tuple[int, int], pos: Tuple[int, int], next_pos: Tuple[int, int]) -> bool:
        """Checks if a position is a bend"""
        prev_row, prev_col = prev_pos
        next_row, next_col = next_pos
        # Get the row and column of the current position
        row, col = pos
        # Check if the current position is a bend
        if (row == prev_row and row == next_row) or (col == prev_col and col == next_col):
            return False
        return True

    def proportion_filled(self) -> float:
        """Returns the proportion of the board that is filled with wires"""
        filled_positions = np.count_nonzero(self.board_layout)
        # Get the total number of positions
        total_positions = self.board_layout.shape[0] * self.board_layout.shape[1]
        # Return the percentage of filled positions
        return filled_positions / total_positions

    def distance_between_heads_and_targets(self) -> List[float]:
        """Returns the L2 distance between the heads and targets of the wires"""
        distances = []
        for head, target in zip(self.heads, self.targets):
            distances.append(self.get_distance_between_cells(head, target))
        return distances

    @staticmethod
    def get_distance_between_cells(cell1: Tuple[int, int], cell2: Tuple[int, int]) -> float:
        """Returns the L2 distance between two cells"""
        return ((cell1[0] - cell2[0]) ** 2 + (cell1[1] - cell2[1]) ** 2) ** 0.5

    def remove_wire(self, wire_index: int) -> None:
        """Removes a wire from the board"""
        if wire_index >= len(self.heads):
            raise ValueError(f"Wire index out of range. Only {len(self.heads)} wires on the board.")
        else:
            # Get the head, target and path of the wire
            head, target, path = self.heads[wire_index], self.targets[wire_index], self.paths[wire_index]
            # Remove the wire from the board
            for pos in path:
                self.board_layout[pos[0]][pos[1]] = 0
            # Remove the wire from the list of heads, targets and paths
            self.heads.pop(wire_index)
            self.targets.pop(wire_index)
            self.paths.pop(wire_index)

            assert len(self.heads) == len(self.targets) == len(
                self.paths), "Heads, targets and paths not of equal length"
            assert head not in self.heads, "Head not removed"
            assert target not in self.targets, "target not removed"
            assert path not in self.paths, "Path not removed"

    def get_board_layout(self) -> np.ndarray:
        # Returns the board layout
        return self.board_layout

    def get_board_statistics(self) -> Dict[str, Union[int, float]]:
        """Returns a dictionary of statistics about the board"""
        num_wires = len(self.heads)
        wire_lengths = [self.get_path_length(path) for path in self.paths]
        avg_wire_length = sum(wire_lengths) / num_wires
        wire_bends = [self.count_path_bends(path) for path in self.paths]
        avg_wire_bends = sum(wire_bends) / num_wires
        avg_head_target_distance = sum(self.distance_between_heads_and_targets()) / num_wires
        proportion_filled = self.proportion_filled()

        # Return summary dict
        summary_dict = dict(num_wires=num_wires, wire_lengths=wire_lengths, avg_wire_length=avg_wire_length,
                            wire_bends=wire_bends, avg_wire_bends=avg_wire_bends,
                            avg_head_target_distance=avg_head_target_distance, percent_filled=proportion_filled)

        return summary_dict

    def is_valid_board(self) -> bool:
        """ Return a boolean indicating if the board is valid.  Raise an exception if not. """
        is_valid = True
        if not self.verify_board_size():
            raise IncorrectBoardSizeError
        if self.board.wires_on_board < 0:
            raise NumAgentsOutOfRangeError
        if not self.verify_encodings_range():
            raise EncodingOutOfRangeError
        if not self.verify_number_heads_tails():
            pass
        if not self.verify_wire_validity():
            raise InvalidWireStructureError
        return is_valid

    def verify_board_size(self) -> bool:
        # Verify that the size of a board layout matches the specified dimensions.
        return np.shape(self.board_layout) == (self.board.rows, self.board.cols)

    def verify_encodings_range(self) -> bool:
        # Verify that all the encodings on the board within the range of 0 to 3 * self._wires_on_board.
        wires_only = np.setdiff1d(self.board_layout, np.array([EMPTY]))
        print(f'Max: {np.max(wires_only)}, Min: {np.min(wires_only)}')
        print(f'Wires on board: {self.board.wires_on_board}')
        print(f'Wires only: {wires_only}')
        if self.board.wires_on_board == 0:
            # if no wires, we should have nothing left of the board
            return len(wires_only) == 0
        if np.min(self.board_layout) < 0:
            return False
        if np.max(self.board_layout) > 3 * self.board.wires_on_board:
            print(np.max(self.board_layout))
            print(3 * self.board.wires_on_board)
            print('Epic Fail')
            return False
        return True

    #
    def verify_number_heads_tails(self) -> bool:
        # Verify that each wire has exactly one head and one target.
        wires_only = np.setdiff1d(self.board_layout, np.array([EMPTY]))
        is_valid = True
        for num_wire in range(self.board.wires_on_board):
            heads = np.count_nonzero(wires_only == (num_wire * 3 + POSITION))
            tails = np.count_nonzero(wires_only == (num_wire * 3 + TARGET))
            if heads < 1 or tails < 1:
                is_valid = False
                raise MissingHeadTailError
            if heads > 1 or tails > 1:
                is_valid = False
                raise DuplicateHeadsTailsError
        return is_valid

    def verify_wire_validity(self) -> bool:
        # Verify that each wire has a valid shape,
        # ie, each head/target is connected to one wire cell, and each wire cell is connected to two.
        for row in range(self.rows):
            for col in range(self.cols):
                cell_label = self.board_layout[row, col]
                # Don't check empty cells
                if cell_label > 0:
                    # Check whether the cell is a wiring path or a starting/target cell
                    if self.position_to_cell_type(row, col) == PATH:
                        # Wiring path cells should have two neighbors of the same wire
                        if self.num_wire_neighbors(cell_label, row, col) != 2:
                            print(
                                f"({row},{col}) == {cell_label}, {self.num_wire_neighbors(cell_label, row, col)} neighbors")
                            return False
                    else:
                        # Head and target cells should only have one neighbor of the same wire.
                        if self.num_wire_neighbors(cell_label, row, col) != 1:
                            print(
                                f"HT({row},{col}) == {cell_label}, {self.num_wire_neighbors(cell_label, row, col)} neighbors")
                            return False
        return True

    def num_wire_neighbors(self, cell_label: int, row: int, col: int) -> int:
        """ Return the number of adjacent cells belonging to the same wire.

            Args:
                cell_label (int) : value of the cell to investigate
                row (int)
                col (int) : (row,col) = 2D position of the cell to investigate

                Returns:
                (int) : The number of adjacent cells belonging to the same wire.
        """
        neighbors = 0
        wire_num = self.cell_label_to_wire_num(cell_label)
        min_val = 3 * wire_num + PATH  # PATH=1 is lowest val
        max_val = 3 * wire_num + TARGET  # TARGET=3 is highest val
        if row > 0:
            if min_val <= self.board_layout[row - 1, col] <= max_val:  # same wire above
                neighbors += 1
        if col > 0:
            if min_val <= self.board_layout[row, col - 1] <= max_val:  # same wire to the left
                neighbors += 1
        if row < self.rows - 1:
            if min_val <= self.board_layout[row + 1, col] <= max_val:  # same wire below
                neighbors += 1
        if col < self.cols - 1:
            if min_val <= self.board_layout[row, col + 1] <= max_val:  # same wire to the right
                neighbors += 1
        return neighbors

    def swap_heads_targets(self) -> None:
        """ Randomly swap the head and target of each wire.  Self.board_layout in modified in-place
        """
        # Loop through all the paths on the board
        # Randomly swap the head and target of each wire (and reverse the direction of the wire)
        for path in self.paths:
            p = np.random.rand()
            if p < 0.5:
                # Swap the head and target of the wire
                head_encoding = self.board_layout[path[0][0], path[0][1]]
                target_encoding = self.board_layout[path[-1][0], path[-1][1]]
                self.board_layout[path[0][0], path[0][1]] = target_encoding
                self.board_layout[path[-1][0], path[-1][1]] = head_encoding
                # Reverse the direction of the wire
                path.reverse()

    def shuffle_wire_encodings(self):
        """ Randomly shuffle the encodings of all wires.  Self.board_layout in modified in-place
        """

        # shuffle the indices of the wires and then assign them to the wires in order
        new_indices = list(range(len(self.paths)))
        random.shuffle(new_indices)
        heads = []
        targets = []
        paths = []
        for index, num in enumerate(new_indices):
            heads.append(self.heads[num])
            targets.append(self.targets[num])
            paths.append(self.paths[num])
            # update the encodings of the wires
            for i, pos in enumerate(paths[index]):
                if i == 0:
                    self.board_layout[pos[0], pos[1]] = 3 * index + POSITION
                elif i == len(paths[index]) - 1:
                    self.board_layout[pos[0], pos[1]] = 3 * index + TARGET
                else:
                    self.board_layout[pos[0], pos[1]] = 3 * index + PATH

    def position_to_wire_num(self, row: int, col: int) -> int:
        """ Returns the wire number of the given cell position

            Args:
                row (int): row of the cell
                col (int): column of the cell

            Returns:
                (int) : The wire number that the cell belongs to. Returns -1 if not part of a wire.
        """
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return -1
        else:
            cell_label = self.board_layout[row, col]
            return self.cell_label_to_wire_num(cell_label)

    @staticmethod
    def cell_label_to_wire_num(cell_label: int) -> int:
        """ Returns the wire number of the given cell value

            Args:
                cell_label (int) : the value of the cell in self.layout

            Returns:
                (int) : The wire number that the cell belongs to. Returns -1 if not part of a wire.
        """
        if cell_label == 0:
            return -1
        else:
            return (cell_label - 1) // 3

    def position_to_cell_type(self, row: int, col: int) -> int:
        """
        Return the type of cell at position (row, col) in self.layout
            0 = empty
            1 = path
            2 = position (starting position)
            3 = target

        Args:
            row (int) : The row of the cell
            col (int) : The column of the cell

        Returns:
            (int) : The type of cell (0-3) as detailed above.
        """
        cell = self.board_layout[row, col]
        if cell == 0:
            return cell
        else:
            return ((cell - 1) % 3) + 1
    #
    # def extend_wires(self):
    #     """ Extend the heads and targets of each wire as far as they can go, preference given to current direction.
    #         The implementation is done in-place on self.board_layout
    #     """
    #     prev_layout = None
    #     # Continue as long as the algorithm is still changing the board
    #     while not np.all(prev_layout == self.board_layout):
    #         prev_layout = 1 * self.board_layout
    #         for row in range(self._rows):
    #             for col in range(self._cols):
    #                 # If the cell is not a head or target, ignore it.
    #                 cell_type = self.position_to_cell_type(Position(row, col))
    #                 if (cell_type != STARTING_POSITION) and (cell_type != TARGET):
    #                     continue
    #
    #                 # If we have found a head or target, try to extend it.
    #                 #
    #                 # Get the list of neighbors available to extend to.
    #                 current_pos = Position(row, col)
    #                 poss_extension_list = self.get_open_adjacent_cells(current_pos, [])
    #                 # Convert tuples to Position class
    #                 poss_extension_list = [Position(cell[0], cell[1]) for cell in poss_extension_list]
    #                 # For each possible cell, throw it out if it already touches part of the same wire.
    #                 current_wire_num = self.position_to_wire_num(current_pos)
    #                 for cell in deepcopy(poss_extension_list):  # Use a copy so we can modify the list in the loop
    #                     if self.num_wire_adjacencies(cell, current_wire_num) > 1:
    #                         poss_extension_list.remove(cell)
    #                 # If there is no room to extend, move on.
    #                 if len(poss_extension_list) == 0:
    #                     continue
    #                 # First find the neighboring cell that is part of the same wire, prioritize extending away from it.
    #                 neighbors_list = self.get_neighbors_same_wire(Position(row, col))
    #                 # There should only be one neighbour to choose from for a head or starting_position cell
    #                 neighbor = neighbors_list[0]
    #                 # Try to extend away from previous neighbor
    #                 priority_neighbor = Position(row + (row - neighbor.x), col + (col - neighbor.y))
    #                 # Prioritize extending away from the previous neighbor if possible.
    #                 if priority_neighbor in poss_extension_list:
    #                     self.extend_cell(current_pos, priority_neighbor)
    #                     row, col = min(row, priority_neighbor.x), min(col, priority_neighbor.y)
    #                 else:
    #                     # Otherwise, extend in a random direction
    #                     extension_pos = random.choice(poss_extension_list)
    #                     self.extend_cell(current_pos, extension_pos)
    #                     row, col = min(row, extension_pos.x), min(col, extension_pos.y)
    #     return
    #
    # # This method is used by the extend_wires method
    # def get_neighbors_same_wire(self, pos: Position) -> List:
    #     """ Returns a list of adjacent cells belonging to the same wire.
    #
    #         Args:
    #             pos (Position): 2D position in self.layout
    #
    #         Returns:
    #             (List) : a list of cells (2D positions) adjacent to the queried cell which belong to the same wire
    #     """
    #     output_list = []
    #     wire_num = self.position_to_wire_num(pos)
    #     pos_up = Position(pos.x - 1, pos.y)
    #     pos_down = Position(pos.x + 1, pos.y)
    #     pos_left = Position(pos.x, pos.y - 1)
    #     pos_right = Position(pos.x, pos.y + 1)
    #     if self.position_to_wire_num(pos_up) == wire_num:
    #         output_list.append(pos_up)
    #     if self.position_to_wire_num(pos_down) == wire_num:
    #         output_list.append(pos_down)
    #     if self.position_to_wire_num(pos_left) == wire_num:
    #         output_list.append(pos_left)
    #     if self.position_to_wire_num(pos_right) == wire_num:
    #         output_list.append(pos_right)
    #     return output_list
    #
    # # This method is used by the extend_wires method
    # def num_wire_adjacencies(self, cell: Position, wire_num: int) -> int:
    #     """ Returns the number of cells adjacent to cell which below to the wire specified by wire_num.
    #
    #         Args:
    #             cell (tuple): 2D position in self.board_layout
    #             wire_num (int): Count adjacent contacts with this specified wire.
    #
    #         Returns:
    #             (int) : The number of adjacent cells belonging to the specified wire
    #     """
    #     num_adjacencies = 0
    #     if self.position_to_wire_num(Position(cell.x - 1, cell.y)) == wire_num:
    #         num_adjacencies += 1
    #     if self.position_to_wire_num(Position(cell.x + 1, cell.y)) == wire_num:
    #         num_adjacencies += 1
    #     if self.position_to_wire_num(Position(cell.x, cell.y - 1)) == wire_num:
    #         num_adjacencies += 1
    #     if self.position_to_wire_num(Position(cell.x, cell.y + 1)) == wire_num:
    #         num_adjacencies += 1
    #     return num_adjacencies
    #
    # # This method is used by the extend_wires method
    # def extend_cell(self, current_cell: Position, extension_cell: Position):
    #     """ Extends the head/target of the wire from current_cell to extension_cell
    #
    #         The extension is done in-place on self.board_layout
    #
    #          Args:
    #              current_cell (Position): 2D position of the current head/target cell
    #              extension_cell (Position): 2D position of the cell to extend into.
    #     """
    #     # Extend head/target into new cell
    #     self.board_layout[extension_cell.x, extension_cell.y] = self.board_layout[current_cell.x, current_cell.y]
    #     cell_type = self.position_to_cell_type(current_cell)
    #     # Convert old head/target cell to a wire
    #     self.board_layout[current_cell.x, current_cell.y] += PATH - cell_type
    #     return
    #
    # # This method is used by the extend_wires method
    # def get_open_adjacent_cells(self, input: Position, wire_list: List) -> List:
    #     """ Returns a list of open cells adjacent to the input cell.
    #
    #     Args:
    #         input (Position): The input cell to search adjacent to.
    #         wire_list (List): List of cells already in the wire.
    #
    #     Returns:
    #         List: List of 2D integer tuples, up to four available cells adjacent to the input cell.
    #     """
    #     adjacent_list = []
    #     # Check above, below, to the left and the right and add those cells to the list if available.
    #     if input.x > 0 and self.is_valid_cell(Position(input.x - 1, input.y), wire_list):
    #         adjacent_list.append((input.x - 1, input.y))
    #     if input.y > 0 and self.is_valid_cell(Position(input.x, input.y - 1), wire_list):
    #         adjacent_list.append((input.x, input.y - 1))
    #     if input.x < self.rows - 1 and self.is_valid_cell(Position(input.x + 1, input.y), wire_list):
    #         adjacent_list.append((input.x + 1, input.y))
    #     if input.y < self.cols - 1 and self.is_valid_cell(Position(input.x, input.y + 1), wire_list):
    #         adjacent_list.append((input.x, input.y + 1))
    #     return adjacent_list
    #
    # # This method is used by the get_open_adjacent_cells method, which is used by the extend_wires method
    # def is_valid_cell(self, input: Position, wire_list: List) -> bool:
    #     """ Returns a boolean, true if the cell is valid to add to the wire.
    #
    #          Args:
    #             input (Position): The input cell to investigate.
    #             wire_list (List): List of cells already in the wire.
    #
    #         Returns:
    #             bool: False if the cell is already in use,
    #                   False if the cell connects the wire in a loop.
    #                   True, otherwise.
    #     """
    #     return (self.board_layout[input.x, input.y] == EMPTY) and (input.x, input.y) not in wire_list \
    #         and (self.number_of_adjacent_wires(input, wire_list) < 2)
    #
    # # This method is used by a method used by the extend_wires method
    # def number_of_adjacent_wires(self, input: Position, wire_list: List) -> int:
    #     """ Returns the number of cells adjacent to the input cell which are in the wire_list.
    #
    #     Args:
    #         input (Position): The input cell to search adjacent to.
    #         wire_list (List): List of cells already in the wire.
    #
    #     Returns:
    #         int: Number of adjacent cells that are in the wire_list.
    #     """
    #     num_adjacent = 0
    #     # Check above, below, to the left and the right and count the number in the wire_list.
    #     if (input.x - 1, input.y) in wire_list:
    #         num_adjacent += 1
    #     if (input.x + 1, input.y) in wire_list:
    #         num_adjacent += 1
    #     if (input.x, input.y - 1) in wire_list:
    #         num_adjacent += 1
    #     if (input.x, input.y + 1) in wire_list:
    #         num_adjacent += 1
    #     return num_adjacent
    #

    # def is_adjacent(self, cell_a: Tuple, cell_b: Tuple) -> bool:
    #     """ Return TRUE if the two cells are adjacent, FALSE otherwise
    #
    #        Args:
    #            cell_a (Position) : X,Y position of a cell
    #            cell_b (Position) : X,Y position of another cell
    #
    #        Returns:
    #            bool : True if the cells are adjacent
    #     """
    #     manhattan_distance = abs(cell_a[0] - cell_b[0]) + abs(cell_a[1] - cell_b[1])
    #     return manhattan_distance == 1


def board_processor_tests(n: int, p: Optional[float] = 0) -> None:
    """ Runs a series of tests on the board processors."""
    generator_list = [RandomWalkBoard, BFSBoard, LSystemBoardGen, WFCBoard]
    fill_methods = [None, BFS_fill, LSystem_fill, None]
    for index, generator in enumerate(generator_list):
        start_time = time.time()
        print(generator.__name__)
        summary_dict = {}
        print('Summary of results:')
        for i in range(n):
            board = generator(10, 10, 5)
            if fill_methods[index]:
                fill_methods[index](board)
            if p > random.random():
                print(board.return_solved_board())
            boardprocessor = BoardProcessor(board)
            summary_dict[i] = boardprocessor.get_board_statistics()

        print(f'Time taken: {time.time() - start_time} for {n} boards')
        for key, value in summary_dict[0].items():
            if type(value) != list:
                print(f'{key}: {np.mean([summary_dict[i][key] for i in range(n)])}')

        print('-----------------------')


def BFS_fill(board: BFSBoard) -> None:
    """ Fills the board with a BFS algorithm."""
    test_threshold_dict = {'min_bends': 2, 'min_length': 3}
    clip_nums = [2, 2] * 10
    clip_methods = ['shortest', 'min_bends'] * 10
    board.fill_clip_with_thresholds(clip_nums, clip_methods, verbose=False, threshold_dict=test_threshold_dict)


def LSystem_fill(board: LSystemBoardGen) -> None:
    """ Fills the board with an LSystem algorithm."""
    board.fill(n_steps=10, pushpullnone_ratios=[2, 1, 1])


if __name__ == '__main__':
    # Fill and process 1000 boards
    board_processor_tests(1000)

    # Sample Usage

    # Create a board from a numpy array
    board = np.array(
        [[11, 10, 7, 7, 7, 7, 0, 0, 0, 0], [10, 10, 7, 7, 8, 7, 0, 0, 9, 0], [10, 10, 12, 7, 7, 7, 7, 7, 7, 14],
         [13, 13, 13, 13, 0, 13, 13, 7, 7, 13], [13, 13, 13, 13, 13, 13, 13, 13, 13, 13],
         [13, 15, 4, 6, 4, 4, 4, 13, 13, 0], [0, 0, 4, 4, 4, 0, 4, 0, 0, 0], [1, 1, 3, 1, 1, 1, 4, 5, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 2, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])

    boardprocessor = BoardProcessor(board)

    # Shuffle wire encodings
    boardprocessor.shuffle_wire_encodings()
    print(boardprocessor.get_board_layout())

    # Remove a wire
    boardprocessor.remove_wire(0)
    print(boardprocessor.get_board_layout())

    # Get Board Statistics
    summary_dict = boardprocessor.get_board_statistics()

    # Create a RandomWalkBoard
    board_ = RandomWalkBoard(10, 10, 5)
    print(f'{board_.return_solved_board()}')
    boardprocessor_ = BoardProcessor(board_)

    # Shuffle wire encodings
    boardprocessor_.shuffle_wire_encodings()
    print('Shuffled Wire Encodings')
    print(f'{boardprocessor_.get_board_layout()}')

    # Remove a wire
    boardprocessor_.remove_wire(0)
    print('Removed Wire')
    print(f'{boardprocessor_.get_board_layout()}')

    # Get Board Statistics
    summary_dict_ = boardprocessor.get_board_statistics()
    print(summary_dict_)
