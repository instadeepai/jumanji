# constants has been moved to jumanji/environments/combinatorial/routing/constants.py
# import jumanji.environments.combinatorial.routing.constants
#from jumanji.environments.combinatorial.routing.constants import TARGET, HEAD, EMPTY
#from jumanji.environments.combinatorial.routing.constants import SOURCE as WIRE
HEAD, TARGET, WIRE, EMPTY = 4,3,2,0
# Also available to import from constants OBSTACLE, NOOP, LEFT, LEFT, UP, RIGHT, DOWN
from dataclasses import dataclass
from abstract_board import AbstractBoard
import numpy as np
from copy import deepcopy
import random
from typing import List, Tuple
#from jax.numpy import asarray # Currently jaxlib is not supported on windows.  This will have to be sorted.
#from env_viewer import RoutingViewer # Currently jaxlib is not supported on windows.  This will have to be sorted.

@dataclass#
class Position:
    # Class of 2D tuple of ints, indicating the size of an array or a 2D position or vector.
    x: int
    y: int

class IncorrectBoardSizeError(Exception):
    #Raised when a board size does not match the specified dimensions."
    pass

class NumAgentsOutOfRangeError(Exception):
    "Raised when self._wires_on_board is negative."
    pass
class IndicesOutOfRangeError(Exception):
    "Raised when one or more cells on the board have an invalid index."
    pass

class DuplicateHeadsTailsError(Exception):
    "Raised when one of the heads or tails of a wire is duplicated."
    pass

class MissingHeadTailError(Exception):
    "Raised when one of the heads or tails of a wire is missing."
    pass

class InvalidWireStructureError(Exception):
    "Raised when one or more of the wires has an invalid structure, eg looping or branching."
    pass

class RandomWalkBoard(AbstractBoard):
    """ The boards are 2D np.ndarrays of wiring routes on a printed circuit board.

    The coding of the boards is as follows:
    Empty cells are coded 0.
    Obstacles are coded 1.
    Heads are encoded starting from 4 in multiples of 3: 4, 7, 10, ...
    Targets are encoded starting from 3 in multiples of 3: 3, 6, 9, ...
    Wiring routes connecting the head/target pairs are encoded starting
            at 2 in multiples of 3: 2, 5, 8, ...

    Args:
        rows, cols (int, int) : Dimensions of the board.
        num_agents (int) : Number of wires to attempt to add to the board

    """
    #def __init__(self, rows: int, cols: int):
    def __init__(self, rows: int, cols: int, num_agents:int = 0):
        self._rows = rows
        self._cols = cols
        self._num_agents = num_agents
        self._wires_on_board = 0
        self._dim = Position(rows, cols)
        # Initialize empty board
        self.layout = np.zeros((rows, cols), int)
        # Add wires to the board
        if num_agents is None:
            num_agents = rows * cols  # An impossible target.  Do as many as possible.
        while (self._wires_on_board < self._num_agents) and not self.is_full():
            # board_output.add_wire_start_distance_directions()
            # board_output.add_wire_head_target_erode()
            self.add_wire_random_walk(2 * max(rows, cols))

    def position_to_cell_type(self, position: Position) -> int:
        """
        Return the type of cell at position (row, col) in self.layout
            0 = empty
            1 = obstacle
            2 = wire
            3 = target
            4 = head

        Args:
            row (int), col (int): 2D position of the cell to query.

        Returns:
            (int) : The type of cell (0-4) as detailed above.
        """
        cell = self.layout[position.x, position.y]
        if cell == 0 or cell == 1:
            return cell
        else:
            return ((cell-2) % 3) + 2


    def get_random_head(self) -> Position:
        # Return a random 2D position, a starting point in the array
        rows = random.randint(0, self._dim.x - 1)
        cols = random.randint(0, self._dim.y - 1)
        return Position(rows, cols)

    def get_wiring_directions(self, head: Position) -> Tuple[Position, Position]:
        """ Return two orthogonal directions for the wire to go.

        Args:
            head (Position): The starting point of the wire.

        Returns:
            (Position) : The primary direction to run the wire.
            (Position) : The second direction to run the wire.
        """
        # Point towards the middle of the layout
        if (head.x < self._dim.x/2):
            x_vector = Position(1, 0)
        else:
            x_vector = Position(-1, 0)
        if (head.y < self._dim.y/2):
            y_vector = Position(0, 1)
        else:
            y_vector = Position(0, -1)
        # Randomly assign one of the directions to be primary, the other secondary
        if random.random() > 0.5:
            return x_vector, y_vector
        else:
            return y_vector, x_vector

    def add_wire_start_distance_directions(self) -> None:
        """
        Define a wire and add it to the layout of the board.

        The wire consists of a head cell, a target cell, and any connecting wire.
        This implementation generates the wire using a random starting point,
        then proceeds the given number of steps in the two principle directions.
        """
        invalid_head = True
        while invalid_head == True:
            head = self.get_random_head()
            if self.layout[head.x, head.y]:
                invalid_head = True
            else:
                connectible_list = self.connectible_cells(head.x, head.y)
                #print(f"connectible list = {len(connectible_list)} cells")
                #print(connectible_list)
                position = Position(head.x, head.y)
                dir_primary, dir_second = self.get_wiring_directions(head)
                num_steps = max(self._dim.x, self._dim.y)
                for step in range(num_steps):
                    # Check for valid step in primary direction
                    if (0 <= (position.x + dir_primary.x) < self._dim.x) \
                      and (0 <= (position.y + dir_primary.y) < self._dim.y)\
                      and not self.layout[position.x+dir_primary.x, position.y+dir_primary.y]:
                        position.x += dir_primary.x
                        position.y += dir_primary.y
                        self.layout[position.x, position.y] = 3 * self._wires_on_board + WIRE
                        invalid_head = False
                    # Else check for valid step in second direction
                    elif (0 <= position.x + dir_second.x < self._dim.x) \
                      and (0 <= position.y + dir_second.y < self._dim.y)\
                      and not self.layout[position.x+dir_second.x, position.y+dir_second.y]:
                        position.x += dir_second.x
                        position.y += dir_second.y
                        self.layout[position.x, position.y] = 3 * self._wires_on_board + WIRE
                        invalid_head = False
        # Mark the head and target cells.
        # Randomly swap the head and target cells 50% of the time.
        if random.random() > 0.5:
            head, position = position, head
        self.layout[head.x, head.y] = 3 * self._wires_on_board + HEAD
        self.layout[position.x, position.y] = 3 * self._wires_on_board + TARGET
        self._wires_on_board += 1
        return None

    def is_full(self) -> bool:
        # Return a boolean if there is no room to fit any more wires on the board.
        for i in range(self._dim.x):
            for j in range(self._dim.y):
                # Return False if there are any adjacent open spots.
                if (i < self._dim.x-1):
                    if (self.layout[i, j] == EMPTY) and (self.layout[i+1, j] == EMPTY):
                        return False
                if (j < self._dim.y-1):
                    if (self.layout[i, j] == EMPTY) and (self.layout[i, j+1] == EMPTY):
                        return False
        # Return True if there were no adjacent open spots.
        return True

    def add_wire_random_walk(self, max_steps: int) -> None:
        """ Add a wire by picking a random start point and walking randomly.
        Args:
            max_steps (int): The maximum number of steps to take.
        """
        invalid_head = True
        while invalid_head == True:
            #x_head, y_head = random.randint(0, self._dim.x - 1), random.randint(0, self._dim.y - 1)
            head = Position(random.randint(0, self._dim.x - 1), random.randint(0, self._dim.y - 1))
            # Ensure that the start point isn't already in use.
            if self.layout[head.x, head.y] != EMPTY:
                continue
            #Ensure that it has at least one open cell to connect to.
            wire_list = [(head.x, head.y)]
            open_adjacent_cells = self.get_open_adjacent_cells(head, wire_list)
            if len(open_adjacent_cells) > 0:
                invalid_head = False
        # Walk randomly from the head
        #print("head=",head)
        for step in range(max_steps):
            new_cell = random.choice(open_adjacent_cells)
            #print("step=",step, "cell=", new_cell)
            wire_list.append(new_cell)
            position = Position(new_cell[0], new_cell[1])
            open_adjacent_cells = self.get_open_adjacent_cells(position, wire_list)
            # Terminate the wire if we are stuck or about to create a loop.
            if len(open_adjacent_cells) == 0:
                break
        # Mark the wiring cells.
        for cell in wire_list:
            self.layout[cell[0], cell[1]] = 3 * self._wires_on_board + WIRE
        # Mark the head and target cells.
        # Randomly swap the head and target cells 50% of the time.
        if random.random() > 0.5:
            head, position = position, head
        self.layout[head.x, head.y] = 3 * self._wires_on_board + HEAD
        self.layout[position.x, position.y] = 3 * self._wires_on_board + TARGET
        self._wires_on_board += 1
        return None

    def is_valid_cell(self,input: Position, wire_list: List) -> bool:
        """ Returns a boolean, true if the cell is valid to add to the wire.

             Args:
                input (Position): The input cell to investigate.
                wire_list (List): List of cells already in the wire.

            Returns:
                bool: False if the cell is already in use,
                      False if the cell connects the wire in a loop.
                      True, otherwise.
        """
        return (self.layout[input.x, input.y] == EMPTY) and (input.x, input.y) not in wire_list\
            and (self.number_of_adjacent_wires(input, wire_list) < 2)

    def get_open_adjacent_cells(self, input: Position, wire_list: List) -> List:
        """ Returns a list of open cells adjacent to the input cell.

        Args:
            input (Position): The input cell to search adjacent to.
            wire_list (List): List of cells already in the wire.

        Returns:
            List: List of 2D integer tuples, up to four available cells adjacent to the input cell.
        """
        adjacent_list = []
        # Check above, below, to the left and the right and add those cells to the list if available.
        if input.x > 0 and self.is_valid_cell(Position(input.x-1, input.y), wire_list):
            adjacent_list.append((input.x-1, input.y))
        if input.y > 0 and self.is_valid_cell(Position(input.x, input.y-1), wire_list):
            adjacent_list.append((input.x, input.y-1))
        if input.x < self._dim.x-1 and self.is_valid_cell(Position(input.x + 1, input.y), wire_list):
            adjacent_list.append((input.x+1, input.y))
        if input.y < self._dim.y-1 and self.is_valid_cell(Position(input.x, input.y + 1), wire_list):
            adjacent_list.append((input.x, input.y+1))
        return adjacent_list

    def number_of_adjacent_wires(self, input: Position, wire_list: List) -> int:
        """ Returns the number of cells adjacent to the input cell which are in the wire_list.

        Args:
            input (Position): The input cell to search adjacent to.
            wire_list (List): List of cells already in the wire.

        Returns:
            int: Number of adjacent cells that are in the wire_list.
        """
        num_adjacent = 0
        # Check above, below, to the left and the right and count the number in the wire_list.
        if (input.x-1, input.y) in wire_list:
            num_adjacent += 1
        if (input.x+1, input.y) in wire_list:
            num_adjacent += 1
        if (input.x, input.y-1) in wire_list:
            num_adjacent += 1
        if (input.x, input.y+1) in wire_list:
            num_adjacent += 1
        return num_adjacent

    # The next six methods support the add_wire_head_target_erode method.
    # Currently, that method leaves extraneous loops, so it's in-progress.
    # Also, the recursive limit on the connectible_cells errors out after 1000 recursions.
    def connectible_cells(self, x_head: int, y_head: int) -> List:
        """ Return a list of 2D tuples, cells that are connectible to (x_head, y_head).

        Args:
            x_head, y_head (int, int) : 2D position of the cell to connect to.

        Returns:
            List[Tuple[int,int]...] : output list of connected cells.
        """
        connectible_list = []
        self.add_connectible_cell(x_head, y_head, connectible_list)
        return connectible_list

    def add_connectible_cell(self, x_pos: int, y_pos: int, connectible_list: List) -> List:
        """ Add the specified cell to the list, recursively call adjacent cells, and return list.

        Args:
            x_pos, y_pos (int, int) : 2D position of the cell to add to the list.
            connectible_list (List[Tuple[int,int]...] : input list of connected cells.

        Returns:
            List[Tuple[int,int]...] : output list of connected cells.
        """
        if (x_pos, y_pos) in connectible_list:
            return connectible_list
        connectible_list.append((x_pos, y_pos))
        # Recursively add the cells above, to the right, below, and to the left if they're valid and open
        if self.is_available_cell(x_pos + 1, y_pos, connectible_list):
            self.add_connectible_cell(x_pos + 1, y_pos, connectible_list)
        if self.is_available_cell(x_pos, y_pos + 1, connectible_list):
            self.add_connectible_cell(x_pos, y_pos+1, connectible_list)
        if self.is_available_cell(x_pos - 1, y_pos, connectible_list):
            self.add_connectible_cell(x_pos-1, y_pos, connectible_list)
        if self.is_available_cell(x_pos, y_pos - 1, connectible_list):
            self.add_connectible_cell(x_pos, y_pos - 1, connectible_list)
        return connectible_list

    def is_available_cell(self, x_coord, y_coord, connectible_list):
        if x_coord not in range(0, self._dim.x) or y_coord not in range(0, self._dim.y):
            return False
        return (self.layout[x_coord, y_coord] == EMPTY) and ((x_coord, y_coord) not in connectible_list)


    def is_connectible(self, x_head: int, y_head: int, x_target: int, y_target: int) -> bool:
            """ Return a boolean indicating if the two cells are connectible on the board.

        Args:
            x_head, y_head (int, int) : 2D position of one end of the proposed wire.
            x_target, y_target (int, int) : 2D position of the other end of the proposed wire.

        Returns:
            bool : True if the two are connectible on the board.
        """
            return (x_target, y_target) in self.connectible_cells(x_head, y_head)

    def add_wire_head_target_erode(self) -> None:
        # Add a wire by listing all connectible cells then stripping them down to a thin wire.
        invalid_head = True
        while invalid_head:
            # Randomly pick a head until we pick a valid one
            x_head, y_head = random.randint(0, self._dim.x - 1), random.randint(0, self._dim.y - 1)
            if self.layout[x_head, y_head]:
                continue
            connectible_list = self.connectible_cells(x_head, y_head)
            # If it's not connectible to anything, try a new random head
            if len(connectible_list) < 2:
                continue
            #print(f"connectible list = {len(connectible_list)} cells")
            #print(connectible_list)
            invalid_head = False
            # wire_list is a copy of the connectible cells, which will exclude the head and target
            wire_list = deepcopy(connectible_list)
            wire_list.remove((x_head, y_head))
            x_target, y_target = random.choice(wire_list)
            wire_list.remove((x_target, y_target))
            # Remove the extraneous cells until we can't remove any more
            not_done_removing = True
            while not_done_removing:
                not_done_removing = False
                for cell in wire_list:
                    if self.three_sides_empty(cell, connectible_list)\
                            or self.is_extraneous_corner(cell, connectible_list):
                        wire_list.remove(cell)
                        connectible_list.remove(cell)
                        not_done_removing = True
        # Add the wire to the layout
        for cell in wire_list:
            self.layout[cell[0], cell[1]] = 3 * self._wires_on_board + 2
        if random.random() > 0.5:
            (x_head, y_head), (x_target, y_target) = (x_target, y_target), (x_head, y_head)
        self.layout[x_head, y_head] = 3 * self._wires_on_board + HEAD
        self.layout[x_target, y_target] = 3 * self._wires_on_board + TARGET
        self._wires_on_board += 1
        return None

    def three_sides_empty(self, cell: (int,int), connectible_list: List) -> bool:
        """ Return a boolean, true if at least three of the four adjacent cells are unconnected.

           Args:
               cell (int, int) : The cell to be investigated.
               connectible_list (List[Tuple[int,int]...]) : The list of all cells in the wire.

           Returns:
               bool : True if at least three of the four adjacent cells are unconnected,
                    e.g. the cell is an extraneous stub that can be deleted from the list.
        """
        (x, y) = cell
        num_empty = 0
        if (x-1, y) not in connectible_list:
            num_empty +=1
        if (x, y-1) not in connectible_list:
            num_empty +=1
        if (x+1, y) not in connectible_list:
            num_empty +=1
        if (x, y+1) not in connectible_list:
            num_empty +=1
        return num_empty >= 3

    def is_extraneous_corner(self, cell: (int,int), connectible_list: List) -> bool:
        """ Return a boolean indicating if the cell is an extraneous corner that can be removed.

           Args:
               cell (int, int) : The cell to be investigated.
               connectible_list (List[Tuple[int,int]...]) : The list of all cells in the wire.

           Returns:
               bool : True if the cell is an extraneous corner that can be removed,
                    e.g. it has two adjacent empty cells, and the cell in the opposite corner is full.
        """
        # Initialize variables.
        (x,y) = cell
        upper_empty, left_empty, bottom_empty, right_empty = False, False, False, False
        botright_full, botleft_full, upright_full, upleft_full = False, False, False, False
        # Check for empty adjacent cells
        upper_empty = (x - 1, y) not in connectible_list
        bottom_empty = (x + 1, y) not in connectible_list
        left_empty = (x, y - 1) not in connectible_list
        right_empty = (x, y + 1) not in connectible_list
        # Check for full corner
        upleft_full = (x  -1, y - 1) in connectible_list
        upright_full = (x - 1, y + 1) in connectible_list
        botleft_full = (x + 1, y - 1) in connectible_list
        botright_full = (x + 1, y + 1) in connectible_list
        # Check if it's a corner cell we can remove
        # If two neighboring adjacent cells are unconnected, it's a corner
        # If the opposite diagonal is connected, this corner is redundant.
        if (upper_empty and left_empty and botright_full)\
            or (upper_empty and right_empty and botleft_full)\
            or (bottom_empty and left_empty and upright_full)\
            or (bottom_empty and right_empty and upleft_full):
            return True
        else:
            return False

    def return_solved_board(self) -> np.array:
        """
        Return an array of the board self.layout with the connecting wires displayed.

        Args: <none>

        Returns:
            (2D np.ndarray of ints) : self.layout
        """
        return self.layout

    def _is_num_agents_hit(self) -> bool:
        """
        Return a boolean indicating whether we have successfully placed _num_agents wires on the board.

        Args: <none>

        Returns:
            (bool) : does wires_on_board match _wires_on_board?
        """
        return self._num_agents == self._wires_on_board


    def return_training_board(self) -> np.ndarray:
        """
        Return a copy of the board self.layout with the connecting wires zeroed out.

        Args: <none>

        Returns:
            layout_out (2D np.ndarray of ints) : same as self.layout with only heads and targets.
        """
        layout_out = []
        for row_in in self.layout:
            # Zero out any element that is a connecting wire
            row_out = [i * int(i%3 != WIRE) for i in row_in]
            layout_out.append(row_out)
        return np.array(layout_out)

    def is_valid_board(self) -> (bool):
        # Return a boolean indicating if the board is valid.  Raise an exception if not.
        is_valid = True
        if not self.verify_board_size():
            raise IncorrectBoardSizeError
            is_valid = False
        if self._wires_on_board < 0:
            raise NumAgentsOutOfRangeError
            is_valid = False
        if not self.verify_indices_range():
            raise IndicesOutOfRangeError
            is_valid = False
        if not self.verify_number_heads_tails():
            is_valid = False
        if not self.verify_wire_validity():
            raise InvalidWireStructureError
            is_valid = False
        return is_valid

    def verify_board_size(self) -> (bool):
        # Verify that the size of a board layout matches the specified dimensions.
        return np.shape(self.layout) == (self._dim.x, self._dim.y)

    def verify_indices_range(self) -> (bool):
        # Verify that all the indices on the board are either 0 or in the range 2 to 3 * self._wires_on_board + 1.
        wires_only = np.setdiff1d(self.layout, np.array([EMPTY]))
        if self._wires_on_board == 0:
            # if no wires, we should have nothing left of the board
            return (len(wires_only) == 0)
        if np.min(self.layout) < 0:
            return False
        if np.max(self.layout) > 3 * self._wires_on_board + 1:
            return False
        return True

    def verify_number_heads_tails(self) -> (bool):
        # Verify that each wire has exactly one head and one target.
        wires_only = np.setdiff1d(self.layout, np.array([EMPTY]))
        is_valid = True
        for num_wire in range(self._wires_on_board):
            heads = np.count_nonzero(wires_only == (num_wire*3 + HEAD))
            tails = np.count_nonzero(wires_only == (num_wire*3 + TARGET))
            if heads < 1 or tails < 1:
                is_valid = False
                raise MissingHeadTailError
            if heads > 1 or tails > 1:
                is_valid = False
                raise DuplicateHeadsTailsError
        return is_valid

    def verify_wire_validity(self) -> (bool):
        # Verify that each wire has a valid shape,
        # ie, each head/target is connected to one wire cell, and each wire cell is connected to two.
        for row in range(self._dim.x):
            for col in range(self._dim.y):
                cell = self.layout[row, col]
                if cell > 0:
                    # Don't check empty cells
                    if (cell % 3) == WIRE:
                        # Wire cells should have two neighbors of the same wire
                        if self.num_wire_neighbors(cell, row, col) != 2:
                            print(f"({row},{col}) == {cell}, {self.num_wire_neighbors(cell, row, col)} neighbors" )
                            return False
                    else:
                        # Head and target cells should only have one neighbor of the same wire.
                        if self.num_wire_neighbors(cell, row, col) != 1:
                            print(f"HT({row},{col}) == {cell}, {self.num_wire_neighbors(cell, row, col)} neighbors")
                            return False
        return True

    def num_wire_neighbors(self, cell: int, row:int, col: int) -> (int):
        """ Return the number of adjacent cells belonging to the same wire.

            Args:
                cell (int) : value of the cell to investigate
                row, col (int, int) : 2D position of the cell to investigate

                Returns:
                (int) : The number of adjacent cells belonging to the same wire.
        """
        neighbors = 0
        wire_num = (cell-2) // 3 # Members of the same wire will be in the range (3*wire_num+2, 3*wire_num+4)
        min_val = 3 * wire_num + 2
        max_val = 3 * wire_num + 4
        if row > 0:
            if min_val <= self.layout[row-1, col] <= max_val: # same wire above
                neighbors +=1
        if col > 0:
            if min_val <= self.layout[row, col-1] <= max_val: # same wire to the left
                neighbors += 1
        if row < self._dim.x - 1:
            if min_val <= self.layout[row + 1, col] <= max_val: # same wire below
                neighbors += 1
        if col < self._dim.y - 1:
            if min_val <= self.layout[row, col + 1] <= max_val: # same wire to the right
                neighbors += 1
        #if neighbors == 0:
            #print(row,col, cell, "min_val = ",min_val,"max_val =",max_val)
        return neighbors#

    def swap_heads_targets(self):
        """ Randomly swap 50% of the heads with their respective targets.  self.layout is modified in-place."""
        for wire_num in range(self._wires_on_board):
            if random.choice([True, False]):
                head_cell = 3 * wire_num + HEAD
                target_cell = 3 * wire_num + TARGET
                #print(f"swap {head_cell} with {target_cell}")
                for x in range(self._dim.x):
                    for y in range(self._dim.y):
                        if self.layout[x, y] == head_cell:
                            self.layout[x, y] = target_cell
                        elif self.layout[x, y] == target_cell:
                            self.layout[x, y] = head_cell
        return # Nothing returned.  self.layout is modified in-place.

    def swap_wires(self, num:int = None):
        """ Randomly swap the numbering of pairs of wires.  Self.layout in modified in-place

            Args:
                num (int) : number of swaps to perform, defaults to self._wires_on_board
        """
        if self._wires_on_board < 2:
            return
        if num == None:
            num = self._wires_on_board
        #print("num=", num)
        for i in range(num):
            wire_num_a = np.random.randint(self._wires_on_board)
            wire_num_b = np.random.randint(self._wires_on_board)
            #print(f"{i}Swap wire {wire_num_a} with {wire_num_b}, {3*wire_num_a + 2}-{3*wire_num_a + 4}<->{3*wire_num_b + 2}-{3*wire_num_b+4}")
            for x in range(self._dim.x):
                for y in range(self._dim.y):
                    cell = self.layout[x, y]
                    # If cell in wire A, renumber to wire B
                    if (3*wire_num_a + 2) <= cell <= (3*wire_num_a + 4):
                        self.layout[x,y] += 3*(wire_num_b - wire_num_a)
                    # If cell in wire B, renumber to wire A
                    if (3 * wire_num_b + 2) <= cell <= (3 * wire_num_b + 4):
                        self.layout[x, y] += 3 * (wire_num_a - wire_num_b)
        return # Nothing to return.  self.layout is modified in-place

    def position_to_wire_num(self, pos: Position) -> (int):
        """ Returns the wire number of the given cell position

            Args:
                position (Position) : the value of the cell in self.layout

            Returns:
                (int) : The wire number that the cell belongs to. Returns -1 if not part of a wire.
        """
        if (pos.x not in range(self._rows)) or (pos.y not in range(self._cols)):
            return -1
        else:
            cell_label = self.layout[pos.x, pos.y]
            return self.cell_label_to_wire_num(cell_label)

    def cell_label_to_wire_num(self, cell_label: int) -> (int):
        """ Returns the wire number of the given cell value

            Args:
                cell_label (int) : the value of the cell in self.layout

            Returns:
                (int) : The wire number that the cell belongs to. Returns -1 if not part of a wire.
        """
        if cell_label < 2:
            return -1
        else:
            return ((cell_label-2) // 3)


    def count_detours(self, count_current_wire: bool = False)  -> (int):
        """ Returns the number of wires that have to detour around a head or target cell.

            Args:
                count_current_wire (bool): Should we count wires that wrap around their own heads/targets? (default = False)

            Returns:
                (int) : The number of wires that have to detour around a head or target cell.
        """
        num_detours = 0
        for x in range(self._dim.x):
            for y in range(self._dim.y):
                label_current_cell = self.layout[x,y]
                if (label_current_cell < 2) or ((label_current_cell % 3) == WIRE):
                    continue
                current_wire = self.cell_label_to_wire_num(label_current_cell)
                #print("\n",x,y,"wire",current_wire)
                #
                above = self.layout[:x, y]
                #print("above = ", above)
                above = [self.cell_label_to_wire_num(cell_label) for cell_label in above if cell_label != 0]
                if not count_current_wire:
                    above = [wire_num for wire_num in above if wire_num != current_wire]
                #print("above = ", above)
                below = self.layout[x + 1:, y]
                #print("below = ", below)
                below = [self.cell_label_to_wire_num(cell_label) for cell_label in below if cell_label != 0]
                if not count_current_wire:
                    below = [wire_num for wire_num in below if wire_num != current_wire]
                #print("below = ", below)
                common = (set(above) & set(below))
                #print("common items = ",common)
                num_detours += len(common)
                #
                left = self.layout[x, :y].tolist()
                #print("left = ", left)
                left = [self.cell_label_to_wire_num(cell_label) for cell_label in left if cell_label != 0]
                if not count_current_wire:
                    left = [wire_num for wire_num in left if wire_num != current_wire]
                #print("left = ", left)
                right = self.layout[x, y+1 :].tolist()
                #print("right = ", right)
                right = [self.cell_label_to_wire_num(cell) for cell in right if cell != 0]
                if not count_current_wire:
                    right = [wire_num for wire_num in right if wire_num != current_wire]
                #print("right = ",right)
                common = (set(right) & set(left))
                #print("common items = ",common)
                num_detours += len(common)
        #print("num_detours = ", num_detours)
        return num_detours


    def extend_wires(self):
        """ Extend the heads and targets of each wire as far as they can go, prefernce given to current direction.
            The implementation is done in-place on self.layout
        """
        prev_layout = None
        # Continue as long as the algorithm is still changing the board
        while not np.all(prev_layout == self.layout):
            prev_layout = 1 * self.layout
            for row in range(self._rows):
                for col in range(self._cols):
                    # If the cell is not a head or target, ignore it.
                    cell_type = self.position_to_cell_type(Position(row,col))
                    if not (cell_type == HEAD) and not (cell_type == TARGET):
                        continue

                    # If we have found a head or target, try to extend it.
                    #
                    # Get the list of neighbors available to extend to. er
                    current_pos = Position(row, col)
                    poss_extension_list = self.get_open_adjacent_cells(current_pos, [])
                    # Convert tuples to Position class
                    poss_extension_list = [Position(cell[0], cell[1]) for cell in poss_extension_list]
                    # For each possible cell, throw it out if it already touches part of the same wire.
                    current_wire_num = self.position_to_wire_num(current_pos)
                    for cell in deepcopy(poss_extension_list):  # Use a copy so we can modify the list in the loop
                        if self.num_wire_adjacencies(cell, current_wire_num) > 1:
                            #print("Removing cell ",cell," from ",poss_extension_list)
                            poss_extension_list.remove(cell)
                    # If there is no room to extend, move on.
                    if len(poss_extension_list) == 0:
                        continue
                    # First find the neighboring cell that is part of the same wire and prioritize extending away from it.
                    neighbors_list = self.get_neighbors_same_wire(Position(row, col)) # Should be only one neighbor for a head or target
                    neighbor = neighbors_list[0] # There should only be one similar neighbor of a head/target
                    priority_neighbor = Position(row + (row - neighbor.x), col + (col - neighbor.y)) # Try to extend away form previous neighbor
                    # Prioritize extending away from the previous neighbor if possible.
                    if priority_neighbor in poss_extension_list:
                        self.extend_cell(current_pos, priority_neighbor)
                    else:
                        # Othewise, extend in a random direction
                        self.extend_cell(current_pos, random.choice(poss_extension_list))
        return

    def get_neighbors_same_wire(self, pos: Position) -> List:
        """ Returns a list of adjacent cells belonging to the same wire.

            Args:
                pos (Position): 2D position in self.layout

            Returns:
                (List) : a list of cells (2D positions) adacent to the queried cell which belong to the same wire
        """
        output_list = []
        wire_num = self.position_to_wire_num(pos)
        pos_up = Position(pos.x - 1, pos.y)
        pos_down = Position(pos.x + 1, pos.y)
        pos_left = Position(pos.x, pos.y - 1)
        pos_right = Position(pos.x, pos.y + 1)
        if self.position_to_wire_num(pos_up) == wire_num:
            output_list.append(pos_up)
        if self.position_to_wire_num(pos_down) == wire_num:
            output_list.append(pos_down)
        if self.position_to_wire_num(pos_left) == wire_num:
            output_list.append(pos_left)
        if self.position_to_wire_num(pos_right) == wire_num:
            output_list.append(pos_right)
        return output_list

    def num_wire_adjacencies(self, cell: Position, wire_num: int) -> int:
        """ Returns the number of cells adjacent to cell which below to the wire specified by wire_num.

            Args:
                cell (tuple): 2D position in self.layout
                wire_num (int): Count adjacent contacts with this specified wire.

            Returns:
                (int) : The number of adjacent cells belonging to the specified wire
        """
        num_adjacencies = 0
        if self.position_to_wire_num(Position(cell.x-1, cell.y)) == wire_num:
            num_adjacencies += 1
        if self.position_to_wire_num(Position(cell.x+1, cell.y)) == wire_num:
            num_adjacencies += 1
        if self.position_to_wire_num(Position(cell.x, cell.y - 1)) == wire_num:
            num_adjacencies += 1
        if self.position_to_wire_num(Position(cell.x, cell.y + 1)) == wire_num:
            num_adjacencies += 1
        return num_adjacencies

    def extend_cell(self, current_cell: Position, extension_cell: Position):
        """ Extends the head/target of the wire from current_cell to extension_cell

            The extension is done in-place on self.layout

             Args:
                 current_cell (Position): 2D position of the current head/target cell
                 extension_cell (Position): 2D position of the cell to extend into.
        """
        #print("current cell = ", current_cell, extension_cell)
        # Extend head/target into new cell
        self.layout[extension_cell.x, extension_cell.y] = self.layout[current_cell.x, current_cell.y]
        cell_type = self.position_to_cell_type(current_cell)
        # Convert old head/target cell to a wire
        self.layout[current_cell.x, current_cell.y] += WIRE - cell_type
        return


def print_board(board_training: np.ndarray, board_solution: np.ndarray, num_agents: int) -> None:
    """ Print the training and solution boards with labels """
    rows, cols = len(board_training), len(board_training[0])
    print(f"\n{rows}x{cols} BOARD")
    print(num_agents, " wires")
    print(board_training)
    print("Solved board")
    print(board_solution)
    return

def get_detour_stats(rows: int, cols: int, num_agents: int, num_boards: int = 1000):
    """ Print out the detour stats averaged over a specified number of boardsExtends the head/target of the wire from current_cell to extension_cell

        Args:
            rows, col (int), (int): 2D size of the board
            num_agents (int): number of wires to add to the board
            num_boards (int): number of boards to generate from which to average the stats
    """
    print(f"\n{rows} x {cols}: {num_agents}")
    sampled_detours = []
    sampled_detours_exclude = []
    sampled_detours_extend = []
    sampled_detours_extend_exclude = []
    for i in range(num_boards):
        my_board = RandomWalkBoard(rows, cols, num_agents)
        num_detours = my_board.count_detours(count_current_wire=True)
        sampled_detours.append(num_detours)
        num_detours_exclude = my_board.count_detours(count_current_wire=False)
        sampled_detours_exclude.append(num_detours_exclude)
        my_board.extend_wires()
        num_detours_extend = my_board.count_detours(count_current_wire=True)
        sampled_detours_extend.append(num_detours_extend)
        num_detours_extend_exclude = my_board.count_detours(count_current_wire=False)
        sampled_detours_extend_exclude.append(num_detours_extend_exclude)
    sampled_detours = np.array(sampled_detours)
    sampled_detours_exclude = np.array(sampled_detours_exclude)
    mean = sampled_detours.mean()
    mean_exclude = sampled_detours_exclude.mean()
    print("Average detours = ", mean," = ", int(100 * mean / num_agents), "%     ", mean_exclude," = ", int(100*mean_exclude) / num_agents, "%")
    print("After extension")
    sampled_detours_extend = np.array(sampled_detours_extend)
    sampled_detours_extend_exclude = np.array(sampled_detours_extend_exclude)
    mean = sampled_detours_extend.mean()
    mean_exclude = sampled_detours_extend_exclude.mean()
    print("Average detours = ", mean, " = ", int(100 * mean / num_agents), "%     ",\
                        mean_exclude, " = ",int(100 * mean_exclude) / num_agents, "%")

if __name__ == "__main__":
    for num_agents in range(9):
        rows, cols = 5,5
        my_board = RandomWalkBoard(rows, cols, num_agents)
        board_training, board_solution, wires_on_board = my_board.return_training_board(), my_board.return_solved_board(), my_board._wires_on_board
        print_board(board_training, board_solution, wires_on_board)
    # Test bigger boards
    # Test allowing the number of wires to default to max possible
    for num_agents in range(5):
        rows, cols = 10,11
        my_board = RandomWalkBoard(rows, cols, num_agents)
        board_training, board_solution, wires_on_board = my_board.return_training_board(), my_board.return_solved_board(), my_board._wires_on_board
        print_board(board_training, board_solution, wires_on_board)

    rows, cols = 18,18
    num_agents = 10
    my_board = RandomWalkBoard(rows, cols, num_agents)
    board_training, board_solution, wires_on_board = my_board.return_training_board(), my_board.return_solved_board(), my_board._wires_on_board
    print_board(board_training, board_solution, wires_on_board)
    """
    viewer = RoutingViewer(_wires_on_board=_wires_on_board, grid_rows=rows, grid_cols=cols,
                           viewer_width=500, viewer_height=500)
    im_training = f'board_{rows}x{cols}_w_{_wires_on_board}_wires.png'
    im_solution = f'solved_board_{rows}x{cols}_w_{_wires_on_board}_wires.png'
    viewer.render(board_training, save_img=im_training)
    viewer.render(board_solution, save_img=im_solution)
    """

    rows, cols = 20,20
    num_agents = 17
    my_board = RandomWalkBoard(rows, cols, num_agents)
    board_training, board_solution, wires_on_board = my_board.return_training_board(), my_board.return_solved_board(), my_board._wires_on_board
    valid = my_board.is_valid_board()
    print_board(board_training, board_solution, wires_on_board)
    print("\nSwap some Heads and Tails")
    my_board.swap_heads_targets()
    valid = my_board.is_valid_board()
    board_training, board_solution, wires_on_board = my_board.return_training_board(), my_board.return_solved_board(), my_board._wires_on_board
    print_board(board_training, board_solution, wires_on_board)
    print("\nSwap some wires")
    my_board.swap_wires()
    valid = my_board.is_valid_board()
    board_training, board_solution, wires_on_board = my_board.return_training_board(), my_board.return_solved_board(), my_board._wires_on_board
    print_board(board_training, board_solution, wires_on_board)
    print("Number of detours = ",my_board.count_detours(count_current_wire=False),"\n")
    my_board.extend_wires()
    print(my_board.return_solved_board())
    print("Number of detours = ", my_board.count_detours(count_current_wire=False))
    valid = my_board.is_valid_board()

    for i in range(1000):
        rows = random.randint(3,20)
        cols = random.randint(3,20)
        num_agents = random.randint(1,rows)
        my_board = RandomWalkBoard(rows, cols, num_agents)
        valid = my_board.is_valid_board()
        #print(i, valid)
        if not valid:
            print("BAD BOARD")
            print(my_board.layout)

    """ 
    # The following are tests of the board.is_valid_board() function
    # to ensure that it picks up errors
        
    my_board = RandomWalkBoard(4,4,0)
    #print(my_board.layout)
    #print(my_board._dim)
    valid = my_board.is_valid_board()
    #test size, _wires_on_board, indices, headstails, valid shape
    #my_board._dim.x = 3
    #valid = my_board.is_valid_board()
    #my_board._dim = Position(4,5)
    #valid = my_board.is_valid_board()
    #my_board._wires_on_board = -1
    #valid = my_board.is_valid_board()
    #my_board._wires_on_board = 1
    #valid = my_board.is_valid_board()
    #my_board = RandomWalkBoard(4, 4, 2)
    #my_board._wires_on_board -= 1
    #valid = my_board.is_valid_board()
    #my_board = RandomWalkBoard(4, 4, 0)
    #my_board.layout[0,0] = 2
    #valid = my_board.is_valid_board()
    my_board = RandomWalkBoard(4, 4, 1)
    my_board.layout = np.array([[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0]])
    #valid = my_board.is_valid_board()
    my_board.layout = np.array([[3, 2, 0, 0],
                                [0, 2, 2, 0],
                                [0, 0, 2, 0],
                                [0, 0, 4, 0]])
    #valid = my_board.is_valid_board()
    my_board.layout = np.array([[3, 2, 0, 0],
                                [0, 2, 2, 0],
                                [0, 0, 2, 0],
                                [0, 0, 2, 0]])
    #valid = my_board.is_valid_board()
    my_board.layout = np.array([[0, 2, 0, 0],
                                [0, 2, 2, 0],
                                [0, 0, 2, 0],
                                [0, 0, 4, 0]])
    #valid = my_board.is_valid_board()
    my_board.layout = np.array([[3, 2, 0, 0],
                                [0, 2, 2, 0],
                                [0, 0, 2, 0],
                                [0, 0, 2, 3]])
    #valid = my_board.is_valid_board()
    my_board.layout = np.array([[3, 2, 0, 0],
                                [0, 2, 2, 0],
                                [0, 0, 2, 0],
                                [0, 0, 4, 4]])
    #valid = my_board.is_valid_board()
    my_board.layout = np.array([[3, 2, 0, 0],
                                [0, 2, 2, 0],
                                [0, 0, 2, 0],
                                [2, 0, 4, 0]])
    #valid = my_board.is_valid_board()
    my_board.layout = np.array([[3, 0, 0, 0],
                                [0, 2, 2, 0],
                                [0, 0, 2, 0],
                                [0, 0, 4, 0]])
    #valid = my_board.is_valid_board()
    my_board.layout = np.array([[3, 2, 0, 0],
                                [0, 2, 2, 2],
                                [0, 0, 2, 0],
                                [0, 0, 4, 0]])
    #valid = my_board.is_valid_board()
    """

    print("\nTest count detours")
    get_detour_stats(rows=8, cols=8, num_agents=5, num_boards=1000)
    get_detour_stats(rows=8, cols=8, num_agents=10, num_boards=1000)
    print()

    my_board = RandomWalkBoard(8,8,5)
    print(my_board.return_solved_board())
    my_board.extend_wires()
    print(my_board.return_solved_board())


