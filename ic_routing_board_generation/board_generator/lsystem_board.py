import numpy as np
import copy
from collections import deque
import random

from typing import List, SupportsFloat as Numeric


class LSystemBoardGen:
    def __init__(self, rows, cols: int = None, num_agents:int = None) -> None:
        if cols is not None:
            self.rows = rows
            self.cols = cols

        elif isinstance(rows, int):
            self.rows = rows
            self.cols = rows

        elif len(rows) == 2:
            self.rows = rows[0]
            self.cols = rows[1]
        else:
            raise ValueError('Rows and cols must be either int or tuple of length 2.')

        self.agent_count = num_agents
        self.board = self.initialise_starting_board(self.rows, self.cols, self.agent_count)
        self.agents = self.assign_agent_states()

    def initialise_starting_board(self, rows:int, cols:int, agent_count:int)->np.ndarray:
        """Initialise as starting board of shape (rows, cols) with agent_count agents."""
        board = np.zeros(shape=(rows,cols), dtype=int)
        for agent in range(1, 1+agent_count):
            tries = 0
            while(1):
                tries += 1
                if tries == 100:
                    print(f'ERROR - NO BOARD SPACES FOUND AFTER {100} TRIES')
                    break

                idx0 = np.random.randint(0, rows)
                idx1 = np.random.randint(0, cols)
                if board[idx0, idx1] != 0.0:
                    continue
                else:
                    board[idx0, idx1] = agent
                    break
        return board
    

    # def add_one_agent(self, agent_num:int):
        # pass


    def assign_agent_states(self):
        """Construct agent from board info. 

        Each agent has the following attributes: (agent_num:int, deque).
        """
        agents = []
        for agent_num in range(1, 1+self.agent_count):
            pos = np.argwhere(self.board==agent_num)
            agent_deque = deque(pos)
            agents.append([agent_num, agent_deque])
        return agents

    def find_next_pos(self, loc:np.ndarray)->np.ndarray:
        """Find the next empty spot for an agent head/tail."""
        empty_places = np.argwhere(self.board==0)
        empty_place_distances = np.sqrt(np.sum((loc-empty_places)**2, axis=1))
        empty_neighbour_places = np.argwhere(empty_place_distances<1+1/9)
        empty_neighbour_coords = np.take(empty_places, empty_neighbour_places, axis=0)
        return empty_neighbour_coords[:,0,:]

    def push(self, agent_num:int)->None:
        """Pushes (grows) agent `agent_num` in a random direction."""
        agent = self.agents[agent_num-1]
        growth_choice = np.random.randint(-1, 1) # choose to grow from the head or the tail, :TO UPDATE
        nbs = self.find_next_pos(agent[1][growth_choice])

        if len(nbs) == 0:
            return None
        else:
            growth_loc = np.random.randint(0, len(nbs))     # choose a random growth direction
            self.board[tuple(nbs[growth_loc])] = agent[0]   # grow in a chosen direction

            # update deque to account for new growth
            if growth_choice == 0:
                agent[1].appendleft(nbs[growth_loc])
            elif growth_choice == -1:
                agent[1].append(nbs[growth_loc])

    def pull(self, agent_num:int)->None:
        """Pulls (shrinks) the agent in a random direction."""
        agent = self.agents[agent_num-1]
        shrink_choice = np.random.randint(-1, 1) #:TO UPDATE

        if len(agent[1]) == 1:
            return None
        
        else:
            self.board[tuple(agent[1][shrink_choice])] = 0
            if shrink_choice == 0:
                _ = agent[1].popleft()
            elif shrink_choice == -1:
                _ = agent[1].pop()

    def fill(self, n_steps:int=5, pushpullnone_ratios:List[Numeric]=[2, 1, 1])->None:
        """Fill the board from its current state.
        
        Args:
            n_steps: The number of actions to perform per agent for board generation.
                     This is one of push (increase length), pull (decrease length), and none (do nothing).
            pushpullnone_ratios: A weighing indicating how much of each agent action to perform.
        
        Note:
            Further customisability can be achieved by modifying 'push' or 'pull' 
            by making stochasticity non-uniform.
        """
        for _ in range(n_steps):
            for agent in range(1, 1+len(self.agents)):
                action = random.choices(['push', 'pull', 'none'], weights=pushpullnone_ratios)[0]
                if action == 'push':
                    self.push(agent)
                elif action == 'pull':
                    self.pull(agent)
        
        # a check to ensure each worm is at least 2 long
        for agent in self.agents:
            tries = 0
            while len(agent[1]) < 2:            # if the agent deque is less than 2 long,
                self.push(agent[0])             # push it to make it 2 long
                tries += 1
                if tries == 10: # re-initialise the board due to a stuck length-1 node
                    print('re-initialising the board')
                    self.board = self.initialise_starting_board(self.rows, self.cols, self.agent_count)
                    self.fill(n_steps, pushpullnone_ratios)
    
        return None

    def convert_to_jumanji(self, route=True, new_version=True)->np.ndarray:
        """Converts the generated board into a Jumanji-compatible one."""
        jumanji_board = copy.deepcopy(self.board)
        
        if not new_version:
            for agent in self.agents:
                # it _needs_ to be checked that the wires are at least 2 long for this procedure
                jumanji_board[tuple(agent[1][0])] = self.board[tuple(agent[1][0])]*3 + 1 # head
                jumanji_board[tuple(agent[1][-1])] = self.board[tuple(agent[1][-1])]*3   # tail
                agent_body_length = len(agent[1]) - 2                                    # body
                if agent_body_length > 0:
                    for i in range(1, 1+agent_body_length):
                        if route:
                            jumanji_board[tuple(agent[1][i])] = self.board[tuple(agent[1][i])]*3-1 # body
                        else:
                            jumanji_board[tuple(agent[1][i])] = 0
            return jumanji_board
        
        else:
            for agent in self.agents:
                # it _needs_ to be checked that the wires are at least 2 long for this procedure
                jumanji_board[tuple(agent[1][0])] = self.board[tuple(agent[1][0])]*3 - 1 # position
                jumanji_board[tuple(agent[1][-1])] = self.board[tuple(agent[1][-1])]*3   # target
                agent_body_length = len(agent[1]) - 2                                    # path
                if agent_body_length > 0:
                    for i in range(1, 1+agent_body_length):
                        if route:
                            jumanji_board[tuple(agent[1][i])] = self.board[tuple(agent[1][i])]*3-2
                        else:
                            jumanji_board[tuple(agent[1][i])] = 0
            return jumanji_board

    def return_training_board(self)->np.ndarray:
        # self.fill(n_steps=10, pushpullnone_ratios=[2,1,1])
        return self.convert_to_jumanji(route=False)

    def return_solved_board(self)->np.ndarray:
        # self.fill(n_steps=10, pushpullnone_ratios=[2,1,1])
        return self.convert_to_jumanji(route=True)


if __name__ == '__main__':
    # Example usage:
    # Generate a board with 10 rows, 10 columns, 10 wires (num_agents) and with max 10 attempts to place each wire
    board = LSystemBoardGen(rows=10, cols=10, num_agents=10)

    # Fill the board
    board.fill(n_steps=10, pushpullnone_ratios=[2,1,1]) # <- this is where most of the augmenting happens
    print(board.return_training_board())
    print(board.return_solved_board())
    
    # edit specific board wires
    board.push(agent_num=5) # <- causes the 5th wire to expand from either end, if possible
    board.pull(agent_num=1) # <- causes the 1st wire to contract from either end, if possible
    print(board.return_training_board())
    print(board.return_solved_board())
