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

from jumanji.env import Environment
from jumanji.registration import make, register, registered_environments

"""Environment Registration"""

# Game2048 - the game of 2048 with the default board size of 4x4.
register(id="Game2048-v0", entry_point="jumanji.environments:Game2048")

# Minesweeper on a board of size 10x10 with 10 mines.
register(id="Minesweeper-v0", entry_point="jumanji.environments:Minesweeper")

# RubiksCube - the standard Rubik's Cube puzzle with faces of size 3x3.
register(id="RubiksCube-v0", entry_point="jumanji.environments:RubiksCube")
register(
    id="RubiksCube-partly-scrambled-v0",
    entry_point="jumanji.environments:RubiksCube",
    kwargs={"time_limit": 20, "num_scrambles_on_reset": 7},
)

# TODO: register BinPack
register(
    id="BinPack-v1",
    entry_point="jumanji.environments:BinPack",
    kwargs={
        "generator": None,  # defaults to RandomGenerator()
        "obs_num_ems": 50,
        "reward_fn": None,  # defaults to DenseRewardFn()
        "normalize_dimensions": True,
        "debug": False,
        "render_mode": "human",
    },
)

# TODO: register JobShop

# Knapsack problem with 50 randomly generated items, a total budget
# of 12.5, and a dense reward function.
register(id="Knapsack-v1", entry_point="jumanji.environments:Knapsack")

# Cleaner with 10 rows, 10 colums, 3 agents, a time limit of 100, and a random maze generator.
register(id="Cleaner-v0", entry_point="jumanji.environments:Cleaner")

# CVRP with 20 randomly generated nodes, a maximum capacity of 30,
# a maximum demand for each node of 10, and a dense reward function.
register(id="CVRP-v1", entry_point="jumanji.environments:CVRP")

# Maze with 10 rows and 10 columns, a time limit of 100 and a random maze generator.
register(id="Maze-v0", entry_point="jumanji.environments:Maze")

# Snake game on a board of size 12x12 with a time limit of 4000.
register(id="Snake-v1", entry_point="jumanji.environments:Snake")

# TSP with 20 randomly generated cities and a dense reward function.
register(id="TSP-v1", entry_point="jumanji.environments:TSP")
