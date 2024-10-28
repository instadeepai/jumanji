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
from matplotlib import pyplot as plt

from jumanji.environments import LevelBasedForaging
from jumanji.environments.routing.lbf.generator import RandomGenerator

env = LevelBasedForaging(
    generator=RandomGenerator(grid_size=8, num_agents=10, num_food=15, fov=8)
)
key = jax.random.key(0)
state, timestep = jax.jit(env.reset)(key)
env.render(state)
action = env.action_spec.generate_value()
state, timestep = jax.jit(env.step)(state, action)
env.render(state)
plt.savefig("lbf.png")
