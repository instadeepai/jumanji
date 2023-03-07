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

## Environment Registration


# Snake
register(
    id="Snake-v1",
    entry_point="jumanji.environments:Snake",
    kwargs={"num_rows": 12, "num_cols": 12, "time_limit": 4000},
)


# TSP
register(
    id="TSP-v1",
    entry_point="jumanji.environments:TSP",
    kwargs={"num_cities": 20},
)


# Knapsack
register(
    id="Knapsack-v1",
    entry_point="jumanji.environments:Knapsack",
    kwargs={"num_items": 50, "total_budget": 12.5},
)


# BinPack

# Toy instance of the BinPack environment
register(
    id="BinPack-toy-v0",
    entry_point="jumanji.environments:BinPack",
    kwargs={
        "instance_generator_type": "toy",
        "obs_num_ems": 40,
    },
)

# Random Instance Generator
register(
    id="BinPack-rand20-v0",
    entry_point="jumanji.environments:BinPack",
    kwargs={
        "instance_generator_type": "random",
        "max_num_items": 20,
        "max_num_ems": 80,
        "obs_num_ems": 40,
    },
)

register(
    id="BinPack-rand40-v0",
    entry_point="jumanji.environments:BinPack",
    kwargs={
        "instance_generator_type": "random",
        "max_num_items": 40,
        "max_num_ems": 200,
        "obs_num_ems": 60,
    },
)

register(
    id="BinPack-rand100-v0",
    entry_point="jumanji.environments:BinPack",
    kwargs={
        "instance_generator_type": "random",
        "max_num_items": 100,
        "max_num_ems": 300,
        "obs_num_ems": 150,
    },
)

# Rubik's Cube
register(id="RubiksCube-v0", entry_point="jumanji.environments:RubiksCube")
register(
    id="RubiksCube-partly-scrambled-v0",
    entry_point="jumanji.environments:RubiksCube",
    kwargs={"step_limit": 20, "num_scrambles_on_reset": 7},
)

# Minesweeper
register(id="Minesweeper-v0", entry_point="jumanji.environments:Minesweeper")


# CVRP - number of nodes, maximum capacity and maximum demand are derived from [1, 2]
# [1] Kwon Y., Choo J., Kim B., Yoon I., Min S., Gwon Y. (2020). "POMO: Policy Optimization with
# Multiple Optima for Reinforcement Learning".
# [2] Grinsztajn N., Furelos-Blanco D., Barrett T. (2022). "Population-Based Reinforcement
# Learning for Combinatorial Optimization Problems".
register(
    id="CVRP-v1",
    entry_point="jumanji.environments:CVRP",
    kwargs={"num_nodes": 20, "max_capacity": 30, "max_demand": 10},
)


# 2048
register(
    id="Game2048-v0",
    entry_point="jumanji.environments:Game2048",
    kwargs={"board_size": 4},
)

# Maze
register(
    id="Maze-v0",
    entry_point="jumanji.environments:Maze",
    kwargs={"n_rows": 10, "n_cols": 10},
)

# Cleaner
register(
    id="Cleaner-v0",
    entry_point="jumanji.environments:Cleaner",
    kwargs={"grid_width": 10, "grid_height": 10, "num_agents": 5},
)
