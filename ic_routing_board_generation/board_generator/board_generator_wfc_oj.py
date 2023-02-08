import random
from typing import List, Tuple
import numpy as np
from copy import deepcopy

from wfcutils import step, update_entropy, all_valid_choices


ALL_TILES = [(0,0), (1, 0), (1, 90), (2, 0), (2, 90), 
            (2, 180), (2, 270), (3, 0), (3, 90), 
            (3, 180), (3, 270)]

# List of tile indexes, with their connections
ALL_TILES2 = {
    0: set(),
    1: {'top', 'bottom'},
    2: {'left', 'right'},
    3: {'top', 'left'},
    4: {'top', 'right'},
    5: {'bottom', 'right'},
    6: {'bottom', 'left'},
    7: {'top'},
    8: {'right'},
    9: {'bottom'},
    10: {'left'}
}


TILE_IDX = {
    (0, 0): 0,
    (1, 0): 1,
    (1, 90): 2,
    (2, 0): 3,
    (2, 90): 4,
    (2, 180): 5,
    (2, 270): 6,
    (3, 0): 7,
    (3, 90): 8,
    (3, 180): 9,
    (3, 270): 10
}

ALL_DIRECTIONS = ['top', 'bottom', 'left', 'right']

REVERSE_DIRECTIONS = {
    'top': 'bottom',
    'bottom': 'top',
    'left': 'right',
    'right': 'left'
}

# For each tile, specifies where it needs a connection
ALL_CONNECTS = {
    (0, 0): set(),
    (1, 0): {'top', 'bottom'},
    (1, 90): {'left', 'right'},
    (2, 0): {'top', 'left'},
    (2, 90): {'top', 'right'},
    (2, 180): {'bottom', 'right'},
    (2, 270): {'bottom', 'left'},
    (3, 0): {'top'},
    (3, 90): {'right'},
    (3, 180): {'bottom'},
    (3, 270): {'left'}
}

class Tile():
    def __init__(self, piece):
        self.piece = piece
        self.idx = TILE_IDX[piece]
        self.neighbours = {
            'top':    set(),
            'bottom': set(),
            'left':   set(),
            'right':  set()
        }
        self.exclusions = {
            'top':    set(),
            'bottom': set(),
            'left':   set(),
            'right':  set()
        }
        
    
    def add_neighbours_exclusions(self):
        # Add the neighbours
        for TILE in ALL_TILES:
            tiley = Tile(TILE)
            # Loop through all possible directions to connect to the tile
            for DIRECTION in ALL_DIRECTIONS:
                # Reverse the directions for the other tile
                REVERSE_DIRECTION = REVERSE_DIRECTIONS[DIRECTION]
                # Check if the tile can connect to the other tile
                if DIRECTION in ALL_CONNECTS[self.piece] and REVERSE_DIRECTION in ALL_CONNECTS[tiley.piece]:
                    # Add the other tile to the neighbours
                    self.neighbours[DIRECTION].add(tiley.piece)
                # Also ok if neither tile trying to connect to the other
                elif DIRECTION not in ALL_CONNECTS[self.piece] and REVERSE_DIRECTION not in ALL_CONNECTS[tiley.piece]:
                    self.neighbours[DIRECTION].add(tiley.piece)
                # Otherwise, add the other tile to the exclusions
                else:
                    self.exclusions[DIRECTION].add(tiley.piece)




            
            
        
    def add_neighbour(self, direction, tile):
        self.neighbours[direction].add(tile.piece)
    
    def remove_neighbour(self, direction, tile):
        self.neighbours[direction].remove(tile.piece)

class Board:
    def __init__(self, x: int, y: int):
        """
        x: width of the board
        y: height of the board
        """
        self.x = x
        self.y = y
        self.grid = [[None for i in range(x)] for j in range(y)]
        # Generate the tile set. This includes how tiles can connect to each other
        self.tile_set_generation()

    
    def tile_set_generation(self):
        """
        For each tile, need to specify type and rotation.
        Empty cells are coded 0.
        Wires are coded 1.
        Turns are coded 2.
        Heads / Targets are encoded 3.

        Rotation is specified in degrees, and is a multiple of 90.
        Returns:
            List of tuples, where each tuple is of the form (type, rotation)
        """
        self.tiles = [Tile(TILE) for TILE in ALL_TILES]
        for tile in self.tiles:
            tile.add_neighbours_exclusions()
    
    def wire_separator(self, final_canvas):
        """ 
        Given a solved board, separate the wires into individual wires.

        Pseudo code:
        1. Whilst there are still wires on the board:
            1.1. Find the first wire
            1.2. Follow the wire until it ends
            1.3. Add the wire to the output board
            1.4. Remove the wire from the input board
        """
        canvas = deepcopy(final_canvas)
        # Initialise the output board
        output_board = np.zeros(shape = (self.y, self.x), dtype = int)
        # Initialise the wire counter
        wire_counter = 0
        # Loop through the board, looking for wires
        while np.any(canvas > 6):
            # Find the first start of a wire
            # This corresponds to values 7, 8, 9, 10
            print(canvas)
            start = tuple(np.argwhere(canvas > 6)[0])
            print("start is", start)
            # Follow the wire until it ends
            wire = self.follow_wire(start, canvas)
            # Add the wire to the output board
            # Change this to be proper values, not just the wire counter
            output_board[start] = 4 + 3 * wire_counter
            canvas[start] = 0
            output_board[wire[-1]] = 3 + 3 * wire_counter
            canvas[wire[-1]] = 0
            wire = wire[1:-1]
            for part in wire:
                output_board[part] = 2 + 3 * wire_counter
                # Remove the wire from the input board
                canvas[part] = 0
            # Increment the wire counter
            wire_counter += 1
            print("wire_counter is", wire_counter)

        return output_board
    
    def follow_wire(self, start, canvas):
        """
        From a given start, follow the wire until it ends.
        Returns:
            List of coordinates of the wire
        """
        # Initialise the wire
        wire = [start]
        # Initialise the current position
        current_position = start
        # Initialise the current direction
        print(canvas[tuple(start)])
        current_direction = tuple(ALL_TILES2[canvas[tuple(start)]])[0]
        # Loop until the wire ends
        while True:
            directions = {
                'top':    (-1, 0),
                'bottom': (1, 0),
                'left':   (0, -1),
                'right':  (0, 1)
            }
            print("current direction is", current_direction)
            print("current piece is", canvas[tuple(current_position)])
            # Find the next position
            next_position = tuple([current_position[i] + directions[current_direction][i] for i in range(2)])
            # Check if the next position is an end point
            print("boobeee", next_position)
            print(canvas[next_position])
            if 7 <= canvas[next_position] <= 10:
                # Add the end point to the wire
                wire.append(next_position)
                # Break the loop
                break
            # Otherwise, add the next position to the wire
            wire.append(next_position)
            # Update the current position
            current_position = next_position
            # Update the current direction
            possible_directions = deepcopy(ALL_TILES2[canvas[next_position]])
            print(canvas[next_position])
            print(possible_directions)
            if current_direction == 'top':
                possible_directions.remove('bottom')
            elif current_direction == 'bottom':
                possible_directions.remove('top')
            elif current_direction == 'left':
                possible_directions.remove('right')
            elif current_direction == 'right':
                possible_directions.remove('left')
            current_direction = list(possible_directions)[0]
        
        return wire





    def wfc(self, seed: int = None):
        cols = self.x
        rows = self.y
        tiles = self.tiles
        tile_idx_list = [tile.idx for tile in tiles]
        history = []
        retract = False
        num_tiles = len(tiles)
        observed = np.zeros(shape = (rows, cols))
        canvas = np.zeros(shape = (rows, cols), dtype = int) - 1
        entropy_board = np.zeros(shape = (rows, cols)) + num_tiles
        choices = {}
        for i in range(rows):
            for j in range(cols):
                choices[(i, j)] = all_valid_choices(i, j, rows, cols, num_tiles)

        info = dict(
            entropy_board = entropy_board,
            observed = observed,
            choices = choices,
            history = history,
            canvas = canvas,
            tiles = tiles,
            rows = rows,
            cols = cols,
            tile_idx_list = tile_idx_list
        )

        info_history = []
        info_history_full = []

        while not np.all(info['observed'] == True):
            info_history.append(deepcopy(info))
            info, retract = step(info)
            info_history_full.append(deepcopy(info))
            #print("info choice is", info['choices'])
            
            while retract:
                #print("retracto baby")
                # undo one step
                last_step = info['history'].pop()
                last_row, last_col, last_choice, valid_choices = last_step
                valid_choices.remove(last_choice)
                if len(valid_choices) > 0:
                    info['choices'][(last_row, last_col)] = valid_choices
                else:
                    info = info_history.pop()
                info, retract = step(info, (last_row, last_col))
                info_history_full.append(deepcopy(info))
                
            entropy_board = update_entropy(choices, rows, cols)
        info_history.append(deepcopy(info))
        canvas = info['canvas']
        print(canvas)
        # Need to separate the individual wires
        output = self.wire_separator(canvas)
        return info, output
        




if __name__ == "__main__":
    board = Board(5, 5)
    info, output = board.wfc()
    print(output)

    tiley = Tile((3,180))
    tiley.add_neighbours_exclusions()
    #print(tiley.exclusions)

    #print("looking at neighbours / exclusions of tile 6")
    tileo = Tile((2, 270))
    tileo.add_neighbours_exclusions()
    #print(tileo.neighbours)
    #print(tileo.exclusions)

"""
   # Correct descriptor here?
    @staticmethod
    def tile_set_generation() -> List[Tuple(int, int)]:
        
        For each tile, need to specify type and rotation.
        Empty cells are coded 0.
        Wires are coded 1.
        Turns are coded 2.
        Heads / Targets are encoded 3.

        Rotation is specified in degrees, and is a multiple of 90.
        Returns:
            List of tuples, where each tuple is of the form (type, rotation)
        
        tile_set = [(0,0), (1, 0), (1, 90), (2, 0), (2, 90), (2, 180), (2, 270), (3, 0), (3, 90), (3, 180), (3, 270)]
        return tile_set

"""