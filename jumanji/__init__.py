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
    id="Snake-6x6-v0",
    entry_point="jumanji.environments:Snake",
    kwargs={"n_rows": 6, "n_cols": 6, "time_limit": 500},
)

register(
    id="Snake-12x12-v0",
    entry_point="jumanji.environments:Snake",
    kwargs={"n_rows": 12, "n_cols": 12, "time_limit": 5000},
)


# TSP
register(
    id="TSP50-v0",
    entry_point="jumanji.environments:TSP",
    kwargs={"problem_size": 50},
)

register(
    id="TSP100-v0",
    entry_point="jumanji.environments:TSP",
    kwargs={"problem_size": 100},
)

register(
    id="TSP150-v0",
    entry_point="jumanji.environments:TSP",
    kwargs={"problem_size": 150},
)

register(
    id="TSP200-v0",
    entry_point="jumanji.environments:TSP",
    kwargs={"problem_size": 200},
)


# Knapsack
register(
    id="Knapsack50-v0",
    entry_point="jumanji.environments:Knapsack",
    kwargs={"problem_size": 50, "total_budget": 25},
)

register(
    id="Knapsack100-v0",
    entry_point="jumanji.environments:Knapsack",
    kwargs={"problem_size": 100, "total_budget": 25},
)

register(
    id="Knapsack200-v0",
    entry_point="jumanji.environments:Knapsack",
    kwargs={"problem_size": 200, "total_budget": 25},
)

register(
    id="Knapsack250-v0",
    entry_point="jumanji.environments:Knapsack",
    kwargs={"problem_size": 250, "total_budget": 25},
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

# Routing
register(
    id="Routing-n3-8x8-v0",
    entry_point="jumanji.environments:Routing",
    kwargs={"rows": 8, "cols": 8, "num_agents": 3, "step_limit": 64},
)

register(
    id="Routing-n4-12x12-v0",
    entry_point="jumanji.environments:Routing",
    kwargs={"rows": 12, "cols": 12, "num_agents": 4, "step_limit": 144},
)

register(
    id="Routing-n5-16x16-v0",
    entry_point="jumanji.environments:Routing",
    kwargs={"rows": 16, "cols": 16, "num_agents": 5, "step_limit": 256},
)


# Connect4
register(
    id="Connect4-v0",
    entry_point="jumanji.environments:Connect4",
)


# CVRP - problem size, maximum capacity and maximum demand are derived from [1, 2]
# [1] Kwon Y., Choo J., Kim B., Yoon I., Min S., Gwon Y. (2020). "POMO: Policy Optimization with
# Multiple Optima for Reinforcement Learning".
# [2] Grinsztajn N., Furelos-Blanco D., Barrett T. (2022). "Population-Based Reinforcement
# Learning for Combinatorial Optimization Problem".
register(
    id="CVRP20-v0",
    entry_point="jumanji.environments:CVRP",
    kwargs={"problem_size": 20, "max_capacity": 30, "max_demand": 10},
)

register(
    id="CVRP50-v0",
    entry_point="jumanji.environments:CVRP",
    kwargs={"problem_size": 50, "max_capacity": 40, "max_demand": 10},
)
register(
    id="CVRP100-v0",
    entry_point="jumanji.environments:CVRP",
    kwargs={"problem_size": 100, "max_capacity": 50, "max_demand": 10},
)

register(
    id="CVRP125-v0",
    entry_point="jumanji.environments:CVRP",
    kwargs={"problem_size": 125, "max_capacity": 55, "max_demand": 10},
)

register(
    id="CVRP150-v0",
    entry_point="jumanji.environments:CVRP",
    kwargs={"problem_size": 150, "max_capacity": 60, "max_demand": 10},
)
