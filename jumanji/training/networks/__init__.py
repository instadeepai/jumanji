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

from jumanji.training.networks.binpack.actor_critic import (
    make_actor_critic_networks_binpack,
)
from jumanji.training.networks.binpack.random import make_random_policy_binpack
from jumanji.training.networks.cvrp.actor_critic import make_actor_critic_networks_cvrp
from jumanji.training.networks.cvrp.random import make_random_policy_cvrp
from jumanji.training.networks.knapsack.actor_critic import (
    make_actor_critic_networks_knapsack,
)
from jumanji.training.networks.knapsack.random import make_random_policy_knapsack
from jumanji.training.networks.routing.actor_critic import (
    make_actor_critic_networks_routing,
)
from jumanji.training.networks.routing.random import make_random_policy_routing
from jumanji.training.networks.rubiks_cube.random import make_random_policy_rubiks_cube
from jumanji.training.networks.snake.actor_critic import (
    make_actor_critic_networks_snake,
)
from jumanji.training.networks.snake.random import make_random_policy_snake
from jumanji.training.networks.tsp.actor_critic import make_actor_critic_networks_tsp
from jumanji.training.networks.tsp.random import make_random_policy_tsp
