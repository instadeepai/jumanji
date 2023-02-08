import os
import sys
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

def check_side(side1, side2):
    ratio = 1.0
    num_pixels = np.prod(side1.shape)
    threshold = ratio * num_pixels
    if np.sum(side1 == side2) >= threshold:
        return True
    elif np.sum(side1[:-1] == side2[1:]) >= threshold:
        return True
    elif np.sum(side1[1:] == side2[:-1]) >= threshold:
        return True


def all_valid_choices(i, j, rows, cols, num_tiles):
    """
    Used to initialise the choices dictionary.
    Also used in reduce_prob to remove invalid choices.
    """
    choices = np.arange(num_tiles).tolist()
    # TODO: Remove some boundary tiles from the choices
    if i == 0:
        choices = [x for x in choices if x not in [1,3,4,7]]
    if i == rows - 1:
        choices = [x for x in choices if x not in [1,5,6,9]]
    if j == 0:
        choices = [x for x in choices if x not in [2,3,6,10]]
    if j == cols - 1:
        choices = [x for x in choices if x not in [2,4,5,8]]
    return choices


def reduce_prob(choices, tiles, row, col, rows, cols, TILE_IDX_LIST):
    neighbor_choices = []
    # Changed this to be a function of the tile
    valid_choices = all_valid_choices(row, col, rows, cols, len(TILE_IDX_LIST))
    # Check the top, bottom, left, right neighbors
    for i, j, direction in [[row-1, col, 'bottom'], [row+1, col, 'top'], [row, col-1, 'right'], [row, col+1, 'left']]:
        exclusion_idx_list = []
        if 0 <= i < rows and 0 <= j < cols:
            print("i, j = ", i, j)
            # Look at every choice for the neighbor
            for tile_idx in choices[(i, j)]:
                tile = tiles[tile_idx]
                exclusion_idx_list.append(tile.exclusions[direction])
        print("exclusion idx list is ", exclusion_idx_list)
        total_num = len(exclusion_idx_list)
        if len(exclusion_idx_list) > 0:
            for idx in TILE_IDX_LIST:
                piece = tiles[idx].piece
                vote = 0
                for exclusion in exclusion_idx_list:
                    # Need to convert to indexes
                    if piece in exclusion:
                        vote += 1
                # If every neighbor has this tile as an exclusion, remove it
                if (vote == total_num) and (idx in valid_choices):
                    print("removing ", idx, " from valid choices")
                    valid_choices.remove(idx)
    if len(valid_choices) == 0:
        return None
    else:
        choices[(row, col)] = valid_choices
        print("this is what reduce prob is saying the choices are: ", choices)
        return choices


def get_min_entropy_coord(entropy_board, observed):
    rows, cols = entropy_board.shape
    min_row, min_col = -1, -1
    min_entropy = 1000
    coord_list = []
    for row in range(rows):
        for col in range(cols):
            if not observed[row, col]:
                if 1 <= entropy_board[row, col] < min_entropy:
                    min_entropy = entropy_board[row, col]
                    coord_list = []
                    coord_list.append((row, col))
                elif 1 <= entropy_board[row, col] == min_entropy:
                    coord_list.append((row, col))
    if len(coord_list) > 0:
        coord_idx = np.random.choice(np.arange(len(coord_list)))
        min_row, min_col = coord_list[coord_idx]
        return min_row, min_col
    else:
        return -1, -1


def update_entropy(choices, rows, cols):
    entropy_board = np.zeros(shape = (rows, cols))
    for row in range(rows):
        for col in range(cols):
            entropy_board[row, col] = len(choices[(row, col)])
    return entropy_board


def step(info, row_col = None):
    entropy_board   = info['entropy_board']
    tile_idx_list   = info['tile_idx_list']
    observed        = info['observed']
    choices         = info['choices']
    history         = info['history']
    canvas          = info['canvas']
    tiles           = info['tiles']
    rows            = info['rows']
    cols            = info['cols']
    print(" doing a step!! \o/")
    if row_col:
        row, col = row_col
    else:
        row, col = get_min_entropy_coord(entropy_board, observed)
    # TODO: change here to weighted random choice, include
    # custom weights for each tile
    # This is just choosing a random tile from the choices
    # However, the choices variable is only based on positional data!
    # Think this is the problem
    state = np.random.choice(choices[(row,  col)])
    history.append((row, col, state, choices[(row,  col)]))
    choices_temp = deepcopy(choices)
    choices_temp[(row, col)] = [state]
    retract = False
    
    # compute new probability for 4 immediate neighbors
    for i, j in [[row-1, col], [row+1, col], [row, col-1], [row, col+1]]:
        if 0 <= i < rows and 0 <= j < cols:
            if not observed[i, j]:
                print("Reducing the prob for i, j: ", i, j)
                attempt = reduce_prob(choices_temp, tiles, i, j, rows, cols, tile_idx_list)
                if attempt:
                    choices_temp = attempt
                else:
                    retract = True
                    break
    
    canvas[row,  col] = state
    observed[row, col] = True
    
    info['entropy_board']   = entropy_board
    info['observed']        = observed
    info['choices']         = choices_temp
    info['history']         = history
    info['canvas']          = canvas
    info['tiles']           = tiles
    
    return info, retract


if __name__ == '__main__':
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


    choices = {(0, 0): [0, 5, 8, 9], (0, 1): [6], (0, 2): [0, 2, 5, 6, 8, 9, 10], (0, 3): [0, 6, 9, 10], 
                (1, 0): [0, 1, 4, 5, 7, 8, 9], (1, 1): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                (1, 2): [0], (1, 3): [0, 1, 3, 6, 7, 9, 10], (2, 0): [0, 1, 4, 5, 7, 8, 9],
                (2, 1): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], (2, 2): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                (2, 3): [0, 1, 3, 6, 7, 9, 10], (3, 0): [8], (3, 1): [0, 2, 3, 4, 7, 8, 10],
                (3, 2): [0, 2, 3, 4, 7, 8, 10], (3, 3): [0, 3, 7, 10]
                }
    tiles = [Tile(TILE) for TILE in ALL_TILES]
    for tile in tiles:
        tile.add_neighbours_exclusions()
    reduce_prob(choices, tiles, 1, 1, 4, 4, np.arange(0,11))