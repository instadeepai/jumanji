import random
from typing import List, Tuple
import numpy as np
from copy import deepcopy

from wfcutils import step, update_entropy


ALL_TILES = [(0,0), (1, 0), (1, 90), (2, 0), (2, 90), 
            (2, 180), (2, 270), (3, 0), (3, 90), 
            (3, 180), (3, 270)]

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
        # Add exclusions for boundary tiles
        self.add_boundary_exclusions()

    
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
    
    def add_boundary_exclusions(self):
        pass




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
                choices[(i, j)] = np.arange(num_tiles).tolist()

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
            
            while retract:
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
        print(info['canvas'])
        canvas = info['canvas']
        print(canvas.shape)
        output = np.zeros(shape = (rows, cols), dtype = int)
        # Convert this into a nice image
        for i in range(rows):
            for j in range(cols):
                element = canvas[i][j]
                if element == 0:
                    output[i,j] = 0
                elif 7 <= element <= 10:
                    output[i,j] = 2
                else:
                    output[i,j] = 1
        print(canvas)
        




if __name__ == "__main__":
    board = Board(4, 4)
    board.wfc()

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