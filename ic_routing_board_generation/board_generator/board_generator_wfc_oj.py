import random
from typing import List, Tuple
import numpy as np
from copy import deepcopy

from ic_routing_board_generation.board_generator.abstract_board import AbstractBoard

class AbstractTile():
    @property
    def get_all_tiles(self):
        all_tiles = [
                        frozenset(),
                        frozenset({'top', 'bottom'}),
                        frozenset({'left', 'right'}),
                        frozenset({'top', 'left'}),
                        frozenset({'top', 'right'}),
                        frozenset({'bottom', 'right'}),
                        frozenset({'bottom', 'left'}),
                        frozenset({'top'}),
                        frozenset({'right'}),
                        frozenset({'bottom'}),
                        frozenset({'left'})
                    ]
        return all_tiles
    
    @property
    def get_all_directions(self):
        return ['top', 'bottom', 'left', 'right'] 

    def get_reverse_direction(self, direction):
        reverse_directions = {
            'top': 'bottom',
            'bottom': 'top',
            'left': 'right',
            'right': 'left'
        }
        return reverse_directions[direction]
    
             
    def add_neighbours_exclusions(self):
        # Add the neighbours
        for candidate in self.get_all_tiles:
            # Loop through all possible directions to connect to the tile
            for direction in self.get_all_directions:
                # Reverse the directions for the other tile
                reverse_direction = self.get_reverse_direction(direction)
                # Check if the tile can connect to the other tile
                if direction in self.connections and reverse_direction in candidate:
                    # Add the other tile to the neighbours
                    self.neighbours[direction].add(candidate)
                # Also ok if neither tile trying to connect to the other
                elif direction not in self.connections and reverse_direction not in candidate:
                    self.neighbours[direction].add(candidate)
                # Otherwise, add the other tile to the exclusions
                else:
                    self.exclusions[direction].add(candidate)


class Tile(AbstractTile):
    def __init__(self, connections):
        """
        Specify a tile by its connections
        """
        self.connections = connections
        self.idx = self.get_all_tiles.index(connections)
        # Specify the pieces that can connect to this tile
        self.neighbours = {
            'top':    set(),
            'bottom': set(),
            'left':   set(),
            'right':  set()
        }
        # Specify the pieces that cannot connect to this tile
        self.exclusions = {
            'top':    set(),
            'bottom': set(),
            'left':   set(),
            'right':  set()
        }
        # Add the neighbours and exclusions
        self.add_neighbours_exclusions()


class WFCUtils():
    def __init__(self):
        self.abstract_tile = AbstractTile()

    def check_side(self, side1, side2):
        ratio = 1.0
        num_pixels = np.prod(side1.shape)
        threshold = ratio * num_pixels
        if np.sum(side1 == side2) >= threshold:
            return True
        elif np.sum(side1[:-1] == side2[1:]) >= threshold:
            return True
        elif np.sum(side1[1:] == side2[:-1]) >= threshold:
            return True


    def all_valid_choices(self, i, j, rows, cols, num_tiles):
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


    def reduce_prob(self, choices, tiles, row, col, rows, cols, TILE_IDX_LIST):
        neighbor_choices = []
        # Changed this to be a function of the tile
        valid_choices = self.all_valid_choices(row, col, rows, cols, len(TILE_IDX_LIST))
        # Check the top, bottom, left, right neighbors
        for i, j, direction in [[row-1, col, 'bottom'], [row+1, col, 'top'], [row, col-1, 'right'], [row, col+1, 'left']]:
            exclusion_idx_list = []
            if 0 <= i < rows and 0 <= j < cols:
                # Look at every choice for the neighbor
                for tile_idx in choices[(i, j)]:
                    tile = Tile(tiles[tile_idx])
                    exclusion_idx_list.append(tile.exclusions[direction])
            total_num = len(exclusion_idx_list)
            if len(exclusion_idx_list) > 0:
                for idx in TILE_IDX_LIST:
                    tile_connections = tiles[idx]
                    vote = 0
                    for exclusion in exclusion_idx_list:
                        # Need to convert to indexes
                        if tile_connections in exclusion:
                            vote += 1
                    # If every neighbor has this tile as an exclusion, remove it
                    if (vote == total_num) and (idx in valid_choices):
                        valid_choices.remove(idx)
        if len(valid_choices) == 0:
            return None
        else:
            choices[(row, col)] = valid_choices
            return choices


    def get_min_entropy_coord(self, entropy_board, observed):
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


    def update_entropy(self, choices, rows, cols):
        entropy_board = np.zeros(shape = (rows, cols))
        for row in range(rows):
            for col in range(cols):
                entropy_board[row, col] = len(choices[(row, col)])
        return entropy_board


    def step(self, info, row_col = None):
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
            row, col = self.get_min_entropy_coord(entropy_board, observed)
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
                    attempt = self.reduce_prob(choices_temp, tiles, i, j, rows, cols, tile_idx_list)
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



class WFCBoard(AbstractBoard):
    def __init__(self, x: int, y: int, num_agents: List[float]):
        """
        x: width of the board
        y: height of the board
        """
        self.x = x
        self.y = y
        self.grid = [[None for i in range(x)] for j in range(y)]
        # Generate the tile set. This includes how tiles can connect to each other
        self.abstract_tile = AbstractTile()
        self.weights = self.generate_weights(x, y, num_agents)
        #self
        self.utils = WFCUtils()
        self.num_agents = num_agents
    
    def generate_weights(self, x, y, num_agents):
        """
        Currently just hard-coding a set of weights for the tiles.

        TODO: make this a function of the board size and number of agents
        """
        weights = [
        6, # empty
        7, 7, # wire
        1, 1, 1, 1, # turn
        0.5, 0.5, 0.5, 0.5 # start / end
        ]
        return weights

    
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
            start = tuple(np.argwhere(canvas > 6)[0])
            # Follow the wire until it ends
            wire = self.follow_wire(start, canvas)
            # Add the wire to the output board
            # Change this to be proper values, not just the wire counter
            output_board[start] = 2 + 3 * wire_counter
            canvas[start] = 0
            output_board[wire[-1]] = 3 + 3 * wire_counter
            canvas[wire[-1]] = 0
            wire = wire[1:-1]
            for part in wire:
                output_board[part] = 1 + 3 * wire_counter
                # Remove the wire from the input board
                canvas[part] = 0
            # Increment the wire counter
            wire_counter += 1

        return output_board, wire_counter
    
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
        current_direction = tuple(self.abstract_tile.get_all_tiles[canvas[tuple(start)]])[0]
        # Loop until the wire ends
        while True:
            directions = {
                'top':    (-1, 0),
                'bottom': (1, 0),
                'left':   (0, -1),
                'right':  (0, 1)
            }
            # Find the next position
            next_position = tuple([current_position[i] + directions[current_direction][i] for i in range(2)])
            # Check if the next position is an end point
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
            possible_directions = set(deepcopy(self.abstract_tile.get_all_tiles[canvas[next_position]]))
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
    
    def remove_wires(self, wired_output, wire_counter):
        """
        TODO: Incorporate Ugo and Randy's fancy removal methods.
        """
        output = deepcopy(wired_output)
        # Loop through the wires
        upper_limit = 3 * self.num_agents
        for i in range(self.x):
            for j in range(self.y):
                if output[i, j] > upper_limit:
                    output[i, j] = 0
        
        return output


    def update_weights(self):
        """
        Idea: decrease the weights corresponding to turns, straight lines, and empty space;
        increase the weights corresponding to start and end points.
        """
        for i in range(len(self.weights)):
            if i < 7:
                self.weights[i] *= 0.8
            else:
                self.weights[i] *= 1.2
        
        return


    def wfc(self, seed: int = None):
        cols = self.x
        rows = self.y
        tiles = AbstractTile().get_all_tiles
        tile_idx_list = list(range(len(tiles)))
        utils = WFCUtils()
        history = []
        retract = False
        num_tiles = len(tiles)
        observed = np.zeros(shape = (rows, cols))
        canvas = np.zeros(shape = (rows, cols), dtype = int) - 1
        entropy_board = np.zeros(shape = (rows, cols)) + num_tiles
        weights = self.weights
        choices = {}
        for i in range(rows):
            for j in range(cols):
                choices[(i, j)] = utils.all_valid_choices(i, j, rows, cols, num_tiles)

        info = dict(
            entropy_board = entropy_board,
            observed = observed,
            choices = choices,
            history = history,
            canvas = canvas,
            tiles = tiles,
            rows = rows,
            cols = cols,
            tile_idx_list = tile_idx_list,
            weights = weights,
        )

        info_history = []
        info_history_full = []

        while not np.all(info['observed'] == True):
            info_history.append(deepcopy(info))
            info, retract = utils.step(info)
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
                info, retract = utils.step(info, (last_row, last_col))
                info_history_full.append(deepcopy(info))
                
            entropy_board = utils.update_entropy(choices, rows, cols)
        info_history.append(deepcopy(info))
        canvas = info['canvas']
        # Need to separate the individual wires
        wired_output, wire_counter = self.wire_separator(canvas)
        unwired_output = np.zeros(shape = (rows, cols))
        # Remove wires to get the number of wires, 
        # change weights and repeat if we have too few wires
        if wire_counter < self.num_agents:
            # Change the weights
            self.update_weights()
            # Repeat
            return self.wfc()
        else:
            # Remove the wires
            wired_output = self.remove_wires(wired_output, wire_counter)
        # Create the unwired output
        for i in range(rows):
            for j in range(cols):
                if wired_output[i, j] % 3 == 1:
                    unwired_output[i, j] = 0
                else:
                    unwired_output[i, j] = wired_output[i,j]
        return info, wired_output, unwired_output, wire_counter
    

    def return_training_board(self) -> np.ndarray:
        _, _, unwired_output, _ = self.wfc()
        return unwired_output
    
    def return_solved_board(self) -> np.ndarray:
        _, wired_output, _, _ = self.wfc()
        return wired_output
        




if __name__ == "__main__":
    # These correspond to the weights we will use to pick tiles
    # Organised by index
    board = WFCBoard(20, 20, 12)
    info, wired_output, unwired_output, wire_counter = board.wfc()
    print(wired_output)
    print(unwired_output)