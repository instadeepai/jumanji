# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np

from jumanji.env import Environment
from jumanji.environments.logic.rubiks_cube import generator as rubik_generator
from jumanji.environments.logic.sudoku import data as sudoku_data
from jumanji.environments.logic.sudoku import generator as sudoku_generator
from jumanji.registration import make, register, registered_environments
from jumanji.version import __version__

"""Environment Registration"""

###
# Logic Environments
###

# Game2048 - the game of 2048 with the default board size of 4x4.
register(id="Game2048-v1", entry_point="jumanji.environments:Game2048")

# GraphColoring - the graph coloring problem with the default graph of
# 20 number of nodes and 0.8 edge probability.
register(id="GraphColoring-v0", entry_point="jumanji.environments:GraphColoring")

# Minesweeper on a board of size 10x10 with 10 mines.
register(id="Minesweeper-v0", entry_point="jumanji.environments:Minesweeper")

# RubiksCube - the standard Rubik's Cube puzzle with faces of size 3x3.
register(id="RubiksCube-v0", entry_point="jumanji.environments:RubiksCube")
# RubiksCube - an easier version of the standard Rubik's Cube puzzle with faces of size 3x3 yet only
# 7 scrambles at reset time, making it technically maximum 7 actions away from the solution.
partly_scrambled_rubiks_cube_generator = rubik_generator.ScramblingGenerator(
    cube_size=3, num_scrambles_on_reset=7
)
register(
    id="RubiksCube-partly-scrambled-v0",
    entry_point="jumanji.environments:RubiksCube",
    kwargs={"time_limit": 20, "generator": partly_scrambled_rubiks_cube_generator},
)

# Sudoku - the standard Sudoku puzzle with grid of size 9x9. By default 10000 puzzles
# of mixed difficulties are set for reset.

register(id="Sudoku-v0", entry_point="jumanji.environments:Sudoku")

# Sudoku - the standard Sudoku puzzle with grid of size 9x9. 1000 puzzles of very-easy
# difficulty (>46 clues) are set for reset.


current_path = os.path.dirname(os.path.abspath(__file__))
sudoku_path = os.path.join(current_path, "environments", "logic", "sudoku", "data")
database_file = sudoku_data.DATABASES["very-easy"]
database = np.load(os.path.join(sudoku_path, database_file))

very_easy_sudoku_generator = sudoku_generator.DatabaseGenerator(database=database)
register(
    id="Sudoku-very-easy-v0",
    entry_point="jumanji.environments:Sudoku",
    kwargs={"generator": very_easy_sudoku_generator},
)


###
# Packing Environments
###

# 3D bin-packing problem with 20 randomly generated items maximum, 40 EMSs maximum that are
# given in the observation.
register(id="BinPack-v2", entry_point="jumanji.environments:BinPack")

# Job-shop scheduling problem with 20 jobs, 10 machines, at most
# 8 operations per job, and a max operation duration of 6 timesteps.
register(id="JobShop-v0", entry_point="jumanji.environments:JobShop")

# Knapsack problem with 50 randomly generated items, a total budget
# of 12.5, and a dense reward function.
register(id="Knapsack-v1", entry_point="jumanji.environments:Knapsack")

# Tetris - the game of tetris with a grid size of 10x10 and a time limit of 400.
register(id="Tetris-v0", entry_point="jumanji.environments:Tetris")


###
# Routing Environments
###

# Cleaner with 10 rows, 10 colums, 3 agents, a time limit of 100, and a random maze generator.
register(id="Cleaner-v0", entry_point="jumanji.environments:Cleaner")

# Connector with grid size of 10 and 10 agents.
register(id="Connector-v2", entry_point="jumanji.environments:Connector")

# MMST with 3 agents, 36 nodes, 72 edges, 4 nodes to connect per agent, and a time limit of 70.
register(id="MMST-v0", entry_point="jumanji.environments:MMST")

# CVRP with 20 randomly generated nodes, a maximum capacity of 30,
# a maximum demand for each node of 10, and a dense reward function.
register(id="CVRP-v1", entry_point="jumanji.environments:CVRP")

# MultiCVRP with 20 randomly generated nodes, a maximum capacity of 60, 2 vehicles,
# a maximum demand for each node of 10, and a dense reward function.
register(id="MultiCVRP-v0", entry_point="jumanji.environments:MultiCVRP")

# Maze with 10 rows and 10 columns, a time limit of 100 and a random maze generator.
register(id="Maze-v0", entry_point="jumanji.environments:Maze")

# RobotWarehouse with a random generator with 2 shelf rows, 3 shelf columns, a column height of 8,
# 4 agents, a sensor range of 1, and a request queue of size 8.
register(id="RobotWarehouse-v0", entry_point="jumanji.environments:RobotWarehouse")

# Snake game on a board of size 12x12 with a time limit of 4000.
register(id="Snake-v1", entry_point="jumanji.environments:Snake")

# TSP with 20 randomly generated cities and a dense reward function.
register(id="TSP-v1", entry_point="jumanji.environments:TSP")
