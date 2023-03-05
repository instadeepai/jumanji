# constants has been moved to jumanji/environments/combinatorial/routing/constants.py
# import jumanji.environments.combinatorial.routing.constants
#from jumanji.environments.combinatorial.routing.constants import TARGET, HEAD, EMPTY
#from jumanji.environments.combinatorial.routing.constants import SOURCE as WIRE
HEAD, TARGET, WIRE, EMPTY = 4,3,2,0
# Also available to import from constants OBSTACLE, NOOP, LEFT, LEFT, UP, RIGHT, DOWN
from dataclasses import dataclass
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
    "Raised when self.num_agents is negative."
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

class Board_rb:
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
        num_agents (int) : Number of wires to add to the board  THIS IS A PROBLEM TO BE RESOLVED

    """
    #def __init__(self, rows: int, cols: int):
    def __init__(self, rows: int, cols: int, num_agents:int = 0): # INITIALIZING A NON-ZERO NUMBER OF AGENTS IS A PROBLEM TO BE RESOLVED
        self.layout = np.zeros((rows, cols), int)
        self.dim = Position(rows, cols)
        #self.num_agents = 0
        self.num_agents = num_agents

    def get_random_head(self) -> Position:
        # Return a random 2D position, a starting point in the array
        rows = random.randint(0, self.dim.x-1)
        cols = random.randint(0, self.dim.y-1)
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
        if (head.x < self.dim.x/2):
            x_vector = Position(1, 0)
        else:
            x_vector = Position(-1, 0)
        if (head.y < self.dim.y/2):
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
                num_steps = max(self.dim.x, self.dim.y)
                for step in range(num_steps):
                    # Check for valid step in primary direction
                    if (0 <= (position.x + dir_primary.x) < self.dim.x) \
                      and (0 <= (position.y + dir_primary.y) < self.dim.y)\
                      and not self.layout[position.x+dir_primary.x, position.y+dir_primary.y]:
                        position.x += dir_primary.x
                        position.y += dir_primary.y
                        self.layout[position.x, position.y] = 3*self.num_agents + WIRE
                        invalid_head = False
                    # Else check for valid step in second direction
                    elif (0 <= position.x + dir_second.x < self.dim.x) \
                      and (0 <= position.y + dir_second.y < self.dim.y)\
                      and not self.layout[position.x+dir_second.x, position.y+dir_second.y]:
                        position.x += dir_second.x
                        position.y += dir_second.y
                        self.layout[position.x, position.y] = 3*self.num_agents + WIRE
                        invalid_head = False
        # Mark the head and target cells.
        # Randomly swap the head and target cells 50% of the time.
        if random.random() > 0.5:
            head, position = position, head
        self.layout[head.x, head.y] = 3*self.num_agents + HEAD
        self.layout[position.x, position.y] = 3*self.num_agents + TARGET
        self.num_agents += 1
        return None

    def is_full(self) -> bool:
        # Return a boolean if there is no room to fit any more wires on the board.
        for i in range(self.dim.x):
            for j in range(self.dim.y):
                # Return False if there are any adjacent open spots.
                if (i < self.dim.x-1):
                    if (self.layout[i, j] == EMPTY) and (self.layout[i+1, j] == EMPTY):
                        return False
                if (j < self.dim.y-1):
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
            #x_head, y_head = random.randint(0, self.dim.x - 1), random.randint(0, self.dim.y - 1)
            head = Position(random.randint(0, self.dim.x - 1), random.randint(0, self.dim.y - 1))
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
            self.layout[cell[0], cell[1]] = 3 * self.num_agents + WIRE
        # Mark the head and target cells.
        # Randomly swap the head and target cells 50% of the time.
        if random.random() > 0.5:
            head, position = position, head
        self.layout[head.x, head.y] = 3 * self.num_agents + HEAD
        self.layout[position.x, position.y] = 3 * self.num_agents + TARGET
        self.num_agents += 1
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
            List: Up to four available cells adjacent to the input cell.
        """
        adjacent_list = []
        # Check above, below, to the left and the right and add those cells to the list if available.
        if input.x > 0 and self.is_valid_cell(Position(input.x-1, input.y), wire_list):
            adjacent_list.append((input.x-1, input.y))
        if input.y > 0 and self.is_valid_cell(Position(input.x, input.y-1), wire_list):
            adjacent_list.append((input.x, input.y-1))
        if input.x < self.dim.x-1 and self.is_valid_cell(Position(input.x+1, input.y), wire_list):
            adjacent_list.append((input.x+1, input.y))
        if input.y < self.dim.y-1 and self.is_valid_cell(Position(input.x, input.y+1), wire_list):
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
        if x_coord not in range(0, self.dim.x) or y_coord not in range(0, self.dim.y):
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
            x_head, y_head = random.randint(0, self.dim.x-1), random.randint(0, self.dim.y-1)
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
            self.layout[cell[0], cell[1]] = 3*self.num_agents + 2
        if random.random() > 0.5:
            (x_head, y_head), (x_target, y_target) = (x_target, y_target), (x_head, y_head)
        self.layout[x_head, y_head] = 3*self.num_agents + HEAD
        self.layout[x_target, y_target] = 3*self.num_agents + TARGET
        self.num_agents += 1
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
        if self.num_agents < 0:
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
        return np.shape(self.layout) == (self.dim.x, self.dim.y)

    def verify_indices_range(self) -> (bool):
        # Verify that all the indices on the board are either 0 or in the range 2 to 3 * self.num_agents + 1.
        wires_only = np.setdiff1d(self.layout, np.array([EMPTY]))
        if self.num_agents == 0:
            # if no wires, we should have nothing left of the board
            return (len(wires_only) == 0)
        if np.min(self.layout) < 0:
            return False
        if np.max(self.layout) > 3 * self.num_agents + 1:
            return False
        return True

    def verify_number_heads_tails(self) -> (bool):
        # Verify that each wire has exactly one head and one target.
        wires_only = np.setdiff1d(self.layout, np.array([EMPTY]))
        is_valid = True
        for num_wire in range(self.num_agents):
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
        for row in range(self.dim.x):
            for col in range(self.dim.y):
                cell = self.layout[row, col]
                if cell > 0:
                    # Don't check empty cells
                    if (cell % 3) == WIRE:
                        # Wire cells should have two neighbors of the same wire
                        if self.num_wire_neighbors(cell, row, col) != 2:
                            #print(f"{row},{col} == {cell}, {self.num_wire_neighbors(cell, row, col)} neighbors" )
                            return False
                    else:
                        # Head and target cells should only have one neighbor of the same wire.
                        if self.num_wire_neighbors(cell, row, col) != 1:
                            #print(f"HT{row},{col} == {cell}, {self.num_wire_neighbors(cell, row, col)} neighbors")
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
        if row < self.dim.x - 1:
            if min_val <= self.layout[row + 1, col] <= max_val: # same wire below
                neighbors += 1
        if col < self.dim.y - 1:
            if min_val <= self.layout[row, col + 1] <= max_val: # same wire to the right
                neighbors += 1
        #if neighbors == 0:
            #print(row,col, cell, "min_val = ",min_val,"max_val =",max_val)
        return neighbors#

    def swap_heads_targets(self):
        """ Randomly swap 50% of the heads with their respective targets.  self.layout is modified in-place."""
        for wire_num in range(num_agents):
            if random.choice([True, False]):
                head_cell = 3 * wire_num + HEAD
                target_cell = 3 * wire_num + TARGET
                #print(f"swap {head_cell} with {target_cell}")
                for x in range(self.dim.x):
                    for y in range(self.dim.y):
                        if self.layout[x, y] == head_cell:
                            self.layout[x, y] = target_cell
                        elif self.layout[x, y] == target_cell:
                            self.layout[x, y] = head_cell
        return # Nothing returned.  self.layout is modified in-place.

    def swap_wires(self, num:int = None):
        """ Randomly swap the numbering of pairs of wires.  Self.layout in modified in-place

            Args:
                num (int) : number of swaps to perform, defaults to self.num_agents
        """
        if self.num_agents < 2:
            return
        if num == None:
            num = self.num_agents
        #print("num=", num)
        for i in range(num):
            wire_num_a = np.random.randint(self.num_agents)
            wire_num_b = np.random.randint(self.num_agents)
            #print(f"{i}Swap wire {wire_num_a} with {wire_num_b}, {3*wire_num_a + 2}-{3*wire_num_a + 4}<->{3*wire_num_b + 2}-{3*wire_num_b+4}")
            for x in range(self.dim.x):
                for y in range(self.dim.y):
                    cell = self.layout[x, y]
                    # If cell in wire A, renumber to wire B
                    if (3*wire_num_a + 2) <= cell <= (3*wire_num_a + 4):
                        self.layout[x,y] += 3*(wire_num_b - wire_num_a)
                    # If cell in wire B, renumber to wire A
                    if (3 * wire_num_b + 2) <= cell <= (3 * wire_num_b + 4):
                        self.layout[x, y] += 3 * (wire_num_a - wire_num_b)
        return # Nothing to return.  self.layout is modified in-place

    def get_wire_num(self, cell_label: int) -> (int):
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
        for x in range(self.dim.x):
            for y in range(self.dim.y):
                cell_label = self.layout[x,y]
                if (cell_label < 2) or ((cell_label % 3) == WIRE):
                    continue
                current_wire = self.get_wire_num(cell_label)
                #print("\n",x,y,"wire",current_wire)
                #
                above = self.layout[:x, y]
                #print("above = ", above)
                above = [self.get_wire_num(cell) for cell in above if cell != 0]
                if not count_current_wire:
                    above = [wire_num for wire_num in above if wire_num != current_wire]
                #print("above = ", above)
                below = self.layout[x + 1:, y]
                #print("below = ", below)
                below = [self.get_wire_num(cell) for cell in below if cell != 0]
                if not count_current_wire:
                    below = [wire_num for wire_num in below if wire_num != current_wire]
                #print("below = ", below)
                common = (set(above) & set(below))
                #print("common items = ",common)
                num_detours += len(common)
                #
                left = self.layout[x, :y].tolist()
                #print("left = ", left)
                left = [self.get_wire_num(cell) for cell in left if cell != 0]
                if not count_current_wire:
                    left = [wire_num for wire_num in left if wire_num != current_wire]
                #print("left = ", left)
                right = self.layout[x, y+1 :].tolist()
                #print("right = ", right)
                right = [self.get_wire_num(cell) for cell in right if cell != 0]
                if not count_current_wire:
                    right = [wire_num for wire_num in right if wire_num != current_wire]
                #print("right = ",right)
                common = (set(right) & set(left))
                #print("common items = ",common)
                num_detours += len(common)
        #print("num_detours = ", num_detours)
        return num_detours



def board_generator_rb(rows: int, cols: int, num_agents: int = None) -> (np.ndarray, np.ndarray, int):
    """ Generate a circuit board of the specified size and number of wires.

    The circuit board will be of dimensions rows by cols.
    The generator will attempt to add the number of wires specified by num_agents.
    The wiring head/target pairs will be generated randomly.
    If num_agents is not specified (or if it is larger than the number of
    wires that the random process is able to fit), then the generator sill fit as
    many wires as possible.

    The coding of the output boards is outlined in the Board class definition.

    Args:
        rows, cols (int, int) : Dimensions of the board.
        num_agents (int) : Number of wiring pairs to attempt.
                Default (None) => Fit as many wiring pairs as possible.

    Returns:
        (Board_rb) : Board class is defined above with attributes .layout, .dim and .num_agents
    """
    # Initialize the board
    board_output = Board_rb(rows, cols)
    # Add wires to the board
    if num_agents is None:
        num_agents = rows * cols # An impossible target.  Do as many as possible.
    for num_wire in range(num_agents):
        if not board_output.is_full():
            #board_output.add_wire_start_distance_directions()
            #board_output.add_wire_head_target_erode()
            board_output.add_wire_random_walk(2*max(rows, cols))
    # Output the training and solution boards and the number of wires
    board_solution = board_output.layout
    board_training = board_output.return_training_board()
    return board_output

def print_board(board_training: np.ndarray, board_solution: np.ndarray, num_agents: int) -> None:
    """ Print the training and solution boards with labels """
    rows, cols = len(board_training), len(board_training[0])
    print(f"\n{rows}x{cols} BOARD")
    print(num_agents, " wires")
    print(board_training)
    print("Solved board")
    print(board_solution)
    return

if __name__ == "__main__":
    for num_agents in range(9):
        rows, cols = 5,5
        my_board = board_generator_rb(rows, cols, num_agents)
        board_training, board_solution, num_agents = my_board.return_training_board(), my_board.layout, my_board.num_agents
        print_board(board_training, board_solution, num_agents)
    # Test bigger boards
    # Test allowing the number of wires to default to max possible
    for num_agents in range(2):
        rows, cols = 10,11
        my_board = board_generator_rb(rows, cols, num_agents)
        board_training, board_solution, num_agents = my_board.return_training_board(), my_board.layout, my_board.num_agents
        print_board(board_training, board_solution, num_agents)

    rows, cols = 18,18
    wires_requested = 10
    my_board = board_generator_rb(rows, cols, wires_requested)
    board_training, board_solution, num_agents = my_board.return_training_board(), my_board.layout, my_board.num_agents
    print_board(board_training, board_solution, num_agents)
    """
    viewer = RoutingViewer(num_agents=num_agents, grid_rows=rows, grid_cols=cols,
                           viewer_width=500, viewer_height=500)
    im_training = f'board_{rows}x{cols}_w_{num_agents}_wires.png'
    im_solution = f'solved_board_{rows}x{cols}_w_{num_agents}_wires.png'
    viewer.render(board_training, save_img=im_training)
    viewer.render(board_solution, save_img=im_solution)
    """

    rows, cols = 20,20
    wires_requested = 17
    my_board = board_generator_rb(rows, cols, wires_requested)
    board_training, board_solution, num_agents = my_board.return_training_board(), my_board.layout, my_board.num_agents
    valid = my_board.is_valid_board()
    print_board(board_training, board_solution, num_agents)
    print("\nSwap some Heads and Tails")
    my_board.swap_heads_targets()
    valid = my_board.is_valid_board()
    board_training, board_solution, num_agents = my_board.return_training_board(), my_board.layout, my_board.num_agents
    print_board(board_training, board_solution, num_agents)
    print("\nSwap some wires")
    my_board.swap_wires()
    valid = my_board.is_valid_board()
    board_training, board_solution, num_agents = my_board.return_training_board(), my_board.layout, my_board.num_agents
    print_board(board_training, board_solution, num_agents)

    for i in range(1000):
        rows = random.randint(3,20)
        cols = random.randint(3,20)
        wires_requested = random.randint(1,rows)
        my_board = board_generator_rb(rows, cols, wires_requested)
        valid = my_board.is_valid_board()
        #print(i, valid)
        if not valid:
            print("BAD BOARD")
            print(my_board.layout)

    """ 
    # The following are tests of the board.is_valid_board() function
    # to ensure that it picks up errors
        
    my_board = board_generator(4,4,0)
    #print(my_board.layout)
    #print(my_board.dim)
    valid = my_board.is_valid_board()
    #size, num_agents, indices, headstails, valid shape
    #my_board.dim.x = 3
    #valid = my_board.is_valid_board()
    #my_board.dim = Position(4,5)
    #valid = my_board.is_valid_board()
    #my_board.num_agents = -1
    #valid = my_board.is_valid_board()
    #my_board.num_agents = 1
    #valid = my_board.is_valid_board()
    #my_board = board_generator(4, 4, 2)
    #my_board.num_agents -= 1
    #valid = my_board.is_valid_board()
    #my_board = board_generator(4, 4, 0)
    #my_board.layout[0,0] = 2
    #valid = my_board.is_valid_board()
    my_board = board_generator(4, 4, 1)
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
    sampled_detours = []
    sampled_detours_exclude = []
    print("8 x 8: 5")
    for i in range(1000):
        my_board = board_generator_rb(8, 8, 5)
        #print(my_board.layout)
        num_detours = my_board.count_detours(count_current_wire = True)
        #print(num_detours)
        sampled_detours.append(num_detours)
        num_detours_exclude = my_board.count_detours(count_current_wire=False)
        sampled_detours_exclude.append(num_detours_exclude)
    sampled_detours = np.array(sampled_detours)
    print("Average detours = ",sampled_detours.mean())
    print("STD = ", sampled_detours.std())
    sampled_detours_exclude = np.array(sampled_detours_exclude)
    print("Excluding current wire")
    print("Average detours = ", sampled_detours_exclude.mean())
    print("STD = ", sampled_detours_exclude.std())

    sampled_detours = []
    sampled_detours = []
    sampled_detours_exclude = []
    print("\n8 x 8: 10")
    for i in range(1000):
        my_board = board_generator_rb(8, 8, 10)
        #print(my_board.layout)
        num_detours = my_board.count_detours(count_current_wire = True)
        #print(num_detours)
        sampled_detours.append(num_detours)
        num_detours_exclude = my_board.count_detours(count_current_wire=False)
        sampled_detours_exclude.append(num_detours_exclude)
    sampled_detours = np.array(sampled_detours)
    print("Average detours = ",sampled_detours.mean())
    print("STD = ", sampled_detours.std())
    sampled_detours_exclude = np.array(sampled_detours_exclude)
    print("Excluding current wire")
    print("Average detours = ", sampled_detours_exclude.mean())
    print("STD = ", sampled_detours_exclude.std())