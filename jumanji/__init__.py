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
    id="Snake6-v0",
    entry_point="jumanji.snake:Snake",
    kwargs={"n_rows": 6, "n_cols": 6, "time_limit": 500},
)

register(
    id="Snake12-v0",
    entry_point="jumanji.snake:Snake",
    kwargs={"n_rows": 12, "n_cols": 12, "time_limit": 5000},
)


# TSP
register(
    id="TSP50-v0",
    entry_point="jumanji.tsp:TSP",
    kwargs={"problem_size": 50},
)

register(
    id="TSP100-v0",
    entry_point="jumanji.tsp:TSP",
    kwargs={"problem_size": 100},
)

register(
    id="TSP150-v0",
    entry_point="jumanji.tsp:TSP",
    kwargs={"problem_size": 150},
)

register(
    id="TSP200-v0",
    entry_point="jumanji.tsp:TSP",
    kwargs={"problem_size": 200},
)


# Knapsack
register(
    id="Knapsack50-v0",
    entry_point="jumanji.knapsack:Knapsack",
    kwargs={"problem_size": 50, "total_budget": 25},
)

register(
    id="Knapsack100-v0",
    entry_point="jumanji.knapsack:Knapsack",
    kwargs={"problem_size": 100, "total_budget": 25},
)

register(
    id="Knapsack200-v0",
    entry_point="jumanji.knapsack:Knapsack",
    kwargs={"problem_size": 200, "total_budget": 25},
)

register(
    id="Knapsack250-v0",
    entry_point="jumanji.knapsack:Knapsack",
    kwargs={"problem_size": 250, "total_budget": 25},
)


# BinPacking
# TODO(alex) to finalize once the generators are merged into develop
register(
    id="BinPack-rand20-v0",
    entry_point="jumanji.binpack:BinPack",
    kwargs={"generator": ..., "obs_num_ems": 40},  # max_num_ems: 80
)

register(
    id="BinPack-rand40-v0",
    entry_point="jumanji.binpack:BinPack",
    kwargs={"generator": ..., "obs_num_ems": 60},  # max_num_ems: 200
)

register(
    id="BinPack-rand100-v0",
    entry_point="jumanji.binpack:BinPack",
    kwargs={"generator": ..., "obs_num_ems": 150},  # max_num_ems: ...
)


# Routing
# TODO(alex) agreeing on standard configs (expected name: Routing-n3-v0)
register(
    id="Routing-n3-v0",
    entry_point="jumanji.pcb_grid:SingleRouting",
    kwargs={},
)


# Connect4
register(
    id="Connect4-v0",
    entry_point="jumanji.connect4:Connect4",
)
