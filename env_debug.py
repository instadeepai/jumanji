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

import random
import jumanji
import jax
import jax.numpy as jnp

#from jumanji.environments.routing.boxoban.env import Boxoban
#from jumanji.environments.routing.boxoban.generator import ToyGenerator

random.seed(40)

#toy_gen = ToyGenerator()
#box_env = Boxoban()

box_env = jumanji.make("Boxoban-v0")

key = jax.random.PRNGKey(0)
state, timestep = box_env.reset(key)

for _ in range(10):
    action = jnp.squeeze(jnp.array([random.randint(0, 4)]))

    state, timestep = box_env.step(state, action)

    print(timestep.reward)
