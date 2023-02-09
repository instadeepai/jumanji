import os
import sys
import time
from copy import deepcopy

#import matplotlib.pyplot as plt
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
            # Look at every choice for the neighbor
            for tile_idx in choices[(i, j)]:
                tile = tiles[tile_idx]
                exclusion_idx_list.append(tile.exclusions[direction])
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
                    valid_choices.remove(idx)
    if len(valid_choices) == 0:
        return None
    else:
        choices[(row, col)] = valid_choices
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
    weights         = info['weights']
    if row_col:
        row, col = row_col
    else:
        row, col = get_min_entropy_coord(entropy_board, observed)
    # TODO: change here to weighted random choice, include
    # custom weights for each tile
    relevant_weights = [weights[tile_idx] for tile_idx in choices[(row, col)]]
    relevant_weights = np.array(relevant_weights) / np.sum(relevant_weights)
    state = np.random.choice(choices[(row,  col)], p = relevant_weights)
    history.append((row, col, state, choices[(row,  col)]))
    choices_temp = deepcopy(choices)
    choices_temp[(row, col)] = [state]
    retract = False
    
    # compute new probability for 4 immediate neighbors
    for i, j in [[row-1, col], [row+1, col], [row, col-1], [row, col+1]]:
        if 0 <= i < rows and 0 <= j < cols:
            if not observed[i, j]:
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
