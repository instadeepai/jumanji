from constants import TARGET, HEAD, EMPTY
from constants import SOURCE as WIRE
# Also available to import from constants OBSTACLE, NOOP, LEFT, LEFT, UP, RIGHT, DOWN
from dataclasses import dataclass
import numpy as np
from copy import deepcopy
import random
from typing import List, Tuple
#from jax.numpy import asarray  Currently jaxlib is not supported on windows.  This will have to be sorted.
#from env_viewer import RoutingViewer  Currently jaxlib is not supported on windows.  This will have to be sorted.

@dataclass
class Position:
    # Class of 2D tuple of ints, indicating the size of an array or a 2D position or vector.
    x: int
    y: int

class Board:
    """ The boards are 2D np.ndarrays of wiring routes on a printed circuit board.

    The coding of the boards is as follows:
    Empty cells are coded 0.
    Obstacles are coded 1.
    Heads are encoded starting from 4 in multiples of 3: 4, 7, 10, ...
    Targets are encoded starting from 3 in multiples of 3: 3, 6, 9, ...
    Wiring routes connecting the head/target pairs are encoded starting
	    at 2 in multiples of 3: 2, 5, 8, ...

    Args:
        x_dim, ydim (int, int) : Dimensions of the board.

    """
    def __init__(self, x_dim: int, y_dim: int):
        self.layout = np.zeros((x_dim, y_dim), int)
        self.dim = Position(x_dim, y_dim)
        self.num_wires = 0

    def get_random_head(self) -> Position:
        # Return a random 2D position, a starting point in the array
        x_dim = random.randint(0, self.dim.x-1) 
        y_dim = random.randint(0, self.dim.y-1) 
        return Position(x_dim, y_dim)

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
                        self.layout[position.x, position.y] = 3*self.num_wires + WIRE 
                        invalid_head = False
                    # Else check for valid step in second direction
                    elif (0 <= position.x + dir_second.x < self.dim.x) \
                      and (0 <= position.y + dir_second.y < self.dim.y)\
                      and not self.layout[position.x+dir_second.x, position.y+dir_second.y]:
                        position.x += dir_second.x
                        position.y += dir_second.y
                        self.layout[position.x, position.y] = 3*self.num_wires + WIRE 
                        invalid_head = False
        # Mark the head and target cells.
        # Randomly swap the head and target cells 50% of the time.
        if random.random() > 0.5:
            head, position = position, head
        self.layout[head.x, head.y] = 3*self.num_wires + HEAD
        self.layout[position.x, position.y] = 3*self.num_wires + TARGET
        self.num_wires += 1
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
            self.layout[cell[0], cell[1]] = 3 * self.num_wires + WIRE
        # Mark the head and target cells.
        # Randomly swap the head and target cells 50% of the time.
        if random.random() > 0.5:
            head, position = position, head
        self.layout[head.x, head.y] = 3 * self.num_wires + HEAD
        self.layout[position.x, position.y] = 3 * self.num_wires + TARGET
        self.num_wires += 1
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
            x_head, y_head (int, int) : cell to connect to.

        Returns:
            List[(int,int)...] : output list of connected cells.
        """
        connectible_list = []
        self.add_connectible_cell(x_head, y_head, connectible_list)
        return connectible_list

    def add_connectible_cell(self, x_pos: int, y_pos: int, connectible_list: List) -> List:
        """ Add the specified cell to the list, recursively call adjacent cells, and return list.

        Args:
            x_pos, y_pos (int, int) : cell to add to the list.
            connectible_list (List[(int,int)...] : input list of connected cells.

        Returns:
            List[(int,int)...] : output list of connected cells.
        """
        if (x_pos, y_pos) in connectible_list:
            return connectible_list
        connectible_list.append((x_pos, y_pos))
        # Recursively add the cells above, to the right, below, and to the left if they're valid and open
        if (x_pos < self.dim.x-1) and (self.layout[x_pos+1, y_pos] == EMPTY) and (x_pos+1, y_pos) not in connectible_list:
            self.add_connectible_cell(x_pos+1, y_pos, connectible_list)
        if (y_pos < self.dim.y-1) and (self.layout[x_pos, y_pos+1] == EMPTY) and (x_pos, y_pos+1) not in connectible_list:
            self.add_connectible_cell(x_pos, y_pos+1, connectible_list)
        if (x_pos > 0) and (self.layout[x_pos-1, y_pos] == EMPTY) and (x_pos-1, y_pos) not in connectible_list:
            self.add_connectible_cell(x_pos-1, y_pos, connectible_list)
        if (y_pos > 0) and (self.layout[x_pos, y_pos-1] == EMPTY) and (x_pos, y_pos-1) not in connectible_list:
            self.add_connectible_cell(x_pos, y_pos-1, connectible_list)
        return connectible_list

    def is_connectible(self, x_head: int, y_head: int, x_target: int, y_target: int) -> bool:
	    """ Return a boolean indicating if the two cells are connectible on the board.

        Args:
            x_head, y_head (int, int) : cell at one end of the proposed wire.
            x_target, y_target (int, int) : cell at the other end of the proposed wire.

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
            print(f"connectible list = {len(connectible_list)} cells")
            print(connectible_list)
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
            self.layout[cell[0], cell[1]] = 3*self.num_wires + 2 
        if random.random() > 0.5:
            (x_head, y_head), (x_target, y_target) = (x_target, y_target), (x_head, y_head)
        self.layout[x_head, y_head] = 3*self.num_wires + HEAD
        self.layout[x_target, y_target] = 3*self.num_wires + TARGET
        self.num_wires += 1
        return None

    def three_sides_empty(self, cell: (int,int), connectible_list: List) -> bool:
        """ Return a boolean, true if at least three of the four adjacent cells are unconnected.

           Args:
               cell (int, int) : The cell to be investigated.
               connectible_list (List[(int,int)...]) : The list of all cells in the wire.

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
               connectible_list (List[(int,int)...]) : The list of all cells in the wire.

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

def remove_connecting_wires(layout_in: np.ndarray) -> np.ndarray:
    """
    Return a copy of the input board layout with the connecting wires zeroed out.

    Args:
        layout_in (2D np.ndarray of ints) : layout specifying heads, targets, and connectors.

    Returns:
        layout_out (2D np.ndarray of ints) : same as layout_in with only heads and targets.
    """
    layout_out = []
    for row_in in layout_in:
        # Zero out any element that is a connecting wire
        row_out = [i * int(i%3 != WIRE) for i in row_in]
        layout_out.append(row_out)
    return np.array(layout_out)


def board_generator(x_dim: int, y_dim: int, target_wires: int = None) -> (np.ndarray, np.ndarray, int):
    """ Generate a circuit board of the specified size and number of wires.

    The circuit board will be of dimensions x_dim by y_dim.
    The generator will attempt to add the number of wires specified by target_wires.
    The wiring head/target pairs will be generated randomly.
    If num_wires is not specified (or if it is larger than the number of
    wires that the random process is able to fit), then the generator sill fit as
    many wires as possible.
	
    The coding of the output boards is outlined in the Board class definition.

    Args:
        x_dim, y_dim (int, int) : Dimensions of the board.
        num_wires (int) : Number of wiring pairs to attempt.
	        Default (None) => Fit as many wiring pairs as possible.

    Returns:
        board_training (np.ndarray of 2D integer tuples (x_dim, y_dim))
            Board design with head/target wiring ports marked.
        board_solution (np.ndarray of 2D integer tuples (x_dim, y_dim))
            Board design with connecting wires and head/target ports marked.
	    num_wires_out (int) : number of wires in the output board design.

    """
    # Initialize the board
    board_output = Board(x_dim, y_dim)
    # Add wires to the board
    if target_wires is None:
        target_wires = x_dim * y_dim # An impossible target.  Do as many as possible.
    for num_wire in range(target_wires):
        if not board_output.is_full():
            #board_output.add_wire_start_distance_directions()
            #board_output.add_wire_head_target_erode()
            board_output.add_wire_random_walk(2*max(x_dim, y_dim))
    # Output the training and solution boards and the number of wires
    board_solution = board_output.layout
    board_training = remove_connecting_wires(board_solution)
    return board_training, board_solution, board_output.num_wires

def print_board(board_training: np.ndarray, board_solution: np.ndarray, num_wires: int) -> None:
    """ Print the training and solution boards with labels """
    x_dim, y_dim = len(board_training), len(board_training[0])
    print(f"\n{x_dim}x{y_dim} BOARD")
    print(num_wires, " wires")
    print(board_training)
    print("Solved board")
    print(board_solution)
    return

if __name__ == "__main__":
    for i in range(9):
        x_dim, y_dim = 5,5
        board_training, board_solution, num_wires = board_generator(x_dim, y_dim, i)
        print_board(board_training, board_solution, num_wires)
    # Test bigger boards
    # Test allowing the number of wires to default to max possible
    for i in range(2):
        x_dim, y_dim = 10,11
        board_training, board_solution, num_wires = board_generator(x_dim, y_dim)
        print_board(board_training, board_solution, num_wires)

    x_dim, y_dim = 18,18
    wires_requested = 10
    board_training, board_solution, num_wires = board_generator(x_dim, y_dim, wires_requested)
    print_board(board_training, board_solution, num_wires)
    """
    viewer = RoutingViewer(num_agents=num_wires, grid_rows=x_dim, grid_cols=y_dim,
                           viewer_width=500, viewer_height=500)
    im_training = f'board_{x_dim}x{y_dim}_w_{num_wires}_wires.png'
    im_solution = f'solved_board_{x_dim}x{y_dim}_w_{num_wires}_wires.png'
    viewer.render(board_training, save_img=im_training)
    viewer.render(board_solution, save_img=im_solution)
    """

    x_dim, y_dim = 20,20
    wires_requested = 17
    board_training, board_solution, num_wires = board_generator(x_dim, y_dim, wires_requested)
    print_board(board_training, board_solution, num_wires)



