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

from typing import Tuple

import chex
import esquilax
import jax
import jax.numpy as jnp

from . import types
from .types import AgentParams


@esquilax.transforms.amap
def update_velocity(
    _k: chex.PRNGKey,
    params: types.AgentParams,
    x: Tuple[chex.Array, types.AgentState],
) -> Tuple[float, float]:
    actions, boid = x
    rotation = actions[0] * params.max_rotate * jnp.pi
    acceleration = actions[1] * params.max_accelerate

    new_heading = (boid.heading + rotation) % (2 * jnp.pi)
    new_speeds = jnp.clip(
        boid.speed + acceleration,
        min=params.min_speed,
        max=params.max_speed,
    )

    return new_heading, new_speeds


@esquilax.transforms.amap
def move(
    _key: chex.PRNGKey, _params: types.AgentParams, x: Tuple[chex.Array, float, float]
) -> chex.Array:
    pos, heading, speed = x
    d_pos = jnp.array([speed * jnp.cos(heading), speed * jnp.sin(heading)])
    return (pos + d_pos) % 1.0


def init_state(
    n: int, params: types.AgentParams, key: chex.PRNGKey
) -> types.AgentState:
    k1, k2, k3 = jax.random.split(key, 3)

    positions = jax.random.uniform(k1, (n, 2))
    speeds = jax.random.uniform(
        k2, (n,), minval=params.min_speed, maxval=params.max_speed
    )
    headings = jax.random.uniform(k3, (n,), minval=0.0, maxval=2.0 * jax.numpy.pi)

    return types.AgentState(
        pos=positions,
        speed=speeds,
        heading=headings,
    )


def update_state(
    key: chex.PRNGKey, params: AgentParams, state: types.AgentState, actions: chex.Array
) -> types.AgentState:
    actions = jax.numpy.clip(actions, min=-1.0, max=1.0)
    headings, speeds = update_velocity(key, params, (actions, state))
    positions = move(key, params, (state.pos, headings, speeds))

    return types.AgentState(
        pos=positions,
        speed=speeds,
        heading=headings,
    )


def view(
    _k: chex.PRNGKey,
    params: Tuple[float, float],
    a: types.AgentState,
    b: types.AgentState,
    *,
    n_view: int,
    i_range: float,
) -> chex.Array:
    view_angle, radius = params
    rays = jnp.linspace(
        -view_angle * jnp.pi,
        view_angle * jnp.pi,
        n_view,
        endpoint=True,
    )
    dx = esquilax.utils.shortest_vector(a.pos, b.pos)
    d = jnp.sqrt(jnp.sum(dx * dx)) / i_range
    phi = jnp.arctan2(dx[1], dx[0]) % (2 * jnp.pi)
    dh = esquilax.utils.shortest_vector(phi, a.heading, 2 * jnp.pi)

    angular_width = jnp.arctan2(radius, d)
    left = dh - angular_width
    right = dh + angular_width

    obs = jnp.where(jnp.logical_and(left < rays, rays < right), d, 1.0)
    return obs
