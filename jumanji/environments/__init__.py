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

import sys

from jumanji.environments.logic import game_2048, minesweeper, rubiks_cube
from jumanji.environments.logic.game_2048.env import Game2048
from jumanji.environments.logic.graph_coloring.env import GraphColoring
from jumanji.environments.logic.minesweeper import Minesweeper
from jumanji.environments.logic.rubiks_cube import RubiksCube
from jumanji.environments.logic.sudoku import Sudoku
from jumanji.environments.packing import bin_pack, job_shop, knapsack, tetris
from jumanji.environments.packing.bin_pack.env import BinPack
from jumanji.environments.packing.job_shop.env import JobShop
from jumanji.environments.packing.knapsack.env import Knapsack
from jumanji.environments.packing.tetris.env import Tetris
from jumanji.environments.routing import (
    cleaner,
    connector,
    cvrp,
    maze,
    mmst,
    multi_cvrp,
    robot_warehouse,
    snake,
    tsp,
)
from jumanji.environments.routing.cleaner.env import Cleaner
from jumanji.environments.routing.connector.env import Connector
from jumanji.environments.routing.cvrp.env import CVRP
from jumanji.environments.routing.maze.env import Maze
from jumanji.environments.routing.mmst.env import MMST
from jumanji.environments.routing.multi_cvrp import MultiCVRP
from jumanji.environments.routing.robot_warehouse.env import RobotWarehouse
from jumanji.environments.routing.snake.env import Snake
from jumanji.environments.routing.tsp.env import TSP


def is_colab() -> bool:
    return "google.colab" in sys.modules


def is_notebook() -> bool:
    return "ipykernel" in sys.modules


# In a notebook, force the Matplotlib backend to ngAgg in order for figures to update
# every time render is called for environments that use Matplotlib
# for rendering. Without this, only the last render result is shown per figure.
try:
    import builtins

    if getattr(builtins, "__IPYTHON__", False):
        import IPython

        if is_colab():
            backend = "inline"
        elif is_notebook():
            backend = "notebook"
        else:
            backend = ""
        IPython.get_ipython().run_line_magic("matplotlib", backend)

except ImportError as exc:
    import warnings

    warnings.warn(f"Error importing IPython: {exc}", stacklevel=2)
