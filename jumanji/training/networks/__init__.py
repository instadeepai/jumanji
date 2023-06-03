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

"""
We assume the networks handle a batch of observations: shape (B, ...).
"""

from jumanji.training.networks.bin_pack.actor_critic import (
    make_actor_critic_networks_bin_pack,
)
from jumanji.training.networks.bin_pack.random import make_random_policy_bin_pack
from jumanji.training.networks.cleaner.actor_critic import (
    make_actor_critic_networks_cleaner,
)
from jumanji.training.networks.cleaner.random import make_random_policy_cleaner
from jumanji.training.networks.connector.actor_critic import (
    make_actor_critic_networks_connector,
)
from jumanji.training.networks.connector.random import make_random_policy_connector
from jumanji.training.networks.cvrp.actor_critic import make_actor_critic_networks_cvrp
from jumanji.training.networks.cvrp.random import make_random_policy_cvrp
from jumanji.training.networks.game_2048.actor_critic import (
    make_actor_critic_networks_game_2048,
)
from jumanji.training.networks.game_2048.random import make_random_policy_game_2048
from jumanji.training.networks.graph_coloring.actor_critic import (
    make_actor_critic_networks_graph_coloring,
)
from jumanji.training.networks.graph_coloring.random import (
    make_random_policy_graph_coloring,
)
from jumanji.training.networks.job_shop.actor_critic import (
    make_actor_critic_networks_job_shop,
)
from jumanji.training.networks.job_shop.random import make_random_policy_job_shop
from jumanji.training.networks.knapsack.actor_critic import (
    make_actor_critic_networks_knapsack,
)
from jumanji.training.networks.knapsack.random import make_random_policy_knapsack
from jumanji.training.networks.maze.actor_critic import make_actor_critic_networks_maze
from jumanji.training.networks.maze.random import make_random_policy_maze
from jumanji.training.networks.minesweeper.actor_critic import (
    make_actor_critic_networks_minesweeper,
)
from jumanji.training.networks.minesweeper.random import make_random_policy_minesweeper
from jumanji.training.networks.mmst.actor_critic import make_actor_critic_networks_mmst
from jumanji.training.networks.mmst.random import make_random_policy_mmst
from jumanji.training.networks.multi_cvrp.actor_critic import (
    make_actor_critic_networks_multicvrp,
)
from jumanji.training.networks.multi_cvrp.random import make_random_policy_multicvrp
from jumanji.training.networks.robot_warehouse.actor_critic import (
    make_actor_critic_networks_robot_warehouse,
)
from jumanji.training.networks.robot_warehouse.random import (
    make_random_policy_robot_warehouse,
)
from jumanji.training.networks.rubiks_cube.actor_critic import (
    make_actor_critic_networks_rubiks_cube,
)
from jumanji.training.networks.rubiks_cube.random import make_random_policy_rubiks_cube
from jumanji.training.networks.snake.actor_critic import (
    make_actor_critic_networks_snake,
)
from jumanji.training.networks.snake.random import make_random_policy_snake
from jumanji.training.networks.sudoku.actor_critic import (
    make_cnn_actor_critic_networks_sudoku,
    make_equivariant_actor_critic_networks_sudoku,
)
from jumanji.training.networks.sudoku.random import make_random_policy_sudoku
from jumanji.training.networks.tetris.actor_critic import (
    make_actor_critic_networks_tetris,
)
from jumanji.training.networks.tetris.random import make_random_policy_tetris
from jumanji.training.networks.tsp.actor_critic import make_actor_critic_networks_tsp
from jumanji.training.networks.tsp.random import make_random_policy_tsp
