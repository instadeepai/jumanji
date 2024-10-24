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

    min_required_cells = num_agents + num_food * 3
    assert (
        grid_size**2 >= min_required_cells
    ), "Grid is too small for this many agents and food items."
    assert (
        grid_size**2
    ) * 0.6 >= min_required_cells, r"Make sure 40% of the grid is empty."


# Example usage
# check_placement_feasibility(8, 10, 15)  # Throws an error


env = LevelBasedForaging(
    generator=RandomGenerator(grid_size=8, num_agents=10, num_food=15, fov=8)
)
key = jax.random.key(0)
state, timestep = jax.jit(env.reset)(key)
env.render(state)
action = env.action_spec.generate_value()
state, timestep = jax.jit(env.step)(state, action)
env.render(state)
