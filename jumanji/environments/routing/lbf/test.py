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

import jax

from jumanji.environments import LevelBasedForaging
from jumanji.environments.routing.lbf.generator import RandomGenerator


def check_placement_feasibility(grid_size: int, num_agents: int, num_food: int) -> None:
    total_cells = (
        grid_size - 2
    ) ** 2  # adjust grid_size to account for non-placable borders
    required_cells = (num_agents + num_food) * 9

    # Ensure the grid has enough cells to potentially place
    # all agents and food without any overlap or adjacency
    assert (
        total_cells >= required_cells
    ), "Grid is too small or too many agents/food items for successful placement."


# Example usage
# check_placement_feasibility(8, 5, 2)  # Throws an error


env = LevelBasedForaging(
    generator=RandomGenerator(grid_size=8, num_agents=5, num_food=2, fov=8)
)
key = jax.random.key(0)
state, timestep = jax.jit(env.reset)(key)
env.render(state)
action = env.action_spec.generate_value()
state, timestep = jax.jit(env.step)(state, action)
env.render(state)
