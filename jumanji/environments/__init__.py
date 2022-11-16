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

from jumanji.environments.combinatorial import binpack, cvrp, knapsack, routing, tsp
from jumanji.environments.combinatorial.binpack.env import BinPack
from jumanji.environments.combinatorial.cvrp.env import CVRP
from jumanji.environments.combinatorial.knapsack.env import Knapsack
from jumanji.environments.combinatorial.routing.env import Routing
from jumanji.environments.combinatorial.tsp.env import TSP
from jumanji.environments.games import connect4, snake
from jumanji.environments.games.connect4.env import Connect4
from jumanji.environments.games.snake.env import Snake


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
