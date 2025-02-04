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

import chex
import jax.numpy as jnp

from jumanji.environments.swarms.search_and_rescue.dynamics import RandomWalk, TargetDynamics
from jumanji.environments.swarms.search_and_rescue.types import TargetState


def test_random_walk_dynamics(key: chex.PRNGKey) -> None:
    n_targets = 50
    vel_max = 0.01
    pos_0 = jnp.full((n_targets, 2), 0.5)

    s0 = TargetState(
        pos=pos_0, vel=jnp.zeros((n_targets, 2)), found=jnp.zeros((n_targets,), dtype=bool)
    )

    dynamics = RandomWalk(acc_std=0.005, vel_max=vel_max)
    assert isinstance(dynamics, TargetDynamics)
    s1 = dynamics(key, s0, 1.0)

    assert isinstance(s1, TargetState)
    assert s1.pos.shape == (n_targets, 2)
    assert s1.vel.shape == (n_targets, 2)
    assert jnp.array_equal(s0.found, s1.found)
    speed = jnp.sqrt(jnp.sum(s1.vel**2, axis=1))
    assert speed.shape == (n_targets,)
    assert jnp.all(speed <= vel_max + 1e-6)
    d_pos = jnp.sqrt(jnp.sum((s0.pos - s1.pos) ** 2, axis=1))
    assert jnp.all(d_pos <= vel_max + 1e-6)
