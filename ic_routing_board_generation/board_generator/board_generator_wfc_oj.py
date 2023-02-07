import random
from typing import List, Tuple

ALL_TILES = [(0,0), (1, 0), (1, 90), (2, 0), (2, 90), 
            (2, 180), (2, 270), (3, 0), (3, 90), 
            (3, 180), (3, 270)]

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
        self.neighbours = {
            'top':    set(),
            'bottom': set(),
            'left':   set(),
            'right':  set()
        }
        # Add neighbours
        # Loop through all possible tiles
    
    def add_neighbours(self):
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
        self.grid = [[Tile((0, 0)) for i in range(x)] for j in range(y)]
        # Hand specify the tile set for now, could change later
        #self.
        self.tile_set = self.tile_set_generation()
    
    @staticmethod
    def tile_set_generation() -> List[Tuple[int, int]]:
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
        initial_set = ALL_TILES
        
        
    
    def adjacency_data(self):
        """
        Returns a list of tuples, where each tuple contains two tiles which can appear next to each other.

        For easy mode, I will start without any rotations
        """

"""
    def wfc(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
        
        self.wfc.run()
"""

if __name__ == "__main__":
    tiley = Tile((0, 0))
    print(tiley.neighbours)
    tiley.add_neighbours()
    print(tiley.neighbours)

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