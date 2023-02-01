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

from jumanji.environments.logic import connect4
from jumanji.environments.logic.connect4.env import Connect4
from jumanji.environments.logic.minesweeper import Minesweeper
from jumanji.environments.logic.rubiks_cube import RubiksCube
from jumanji.environments.packing import binpack, knapsack
from jumanji.environments.packing.binpack.env import BinPack
from jumanji.environments.packing.jobshop.env import JobShop
from jumanji.environments.packing.knapsack.env import Knapsack
from jumanji.environments.routing import connector, cvrp, routing, snake, tsp
from jumanji.environments.routing.cvrp.env import CVRP
from jumanji.environments.routing.routing.env import Routing
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

    warnings.warn(f"Error importing IPython: {exc}")
