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
    _: chex.PRNGKey,
    params: types.AgentParams,
    x: Tuple[chex.Array, types.AgentState],
) -> Tuple[float, float]:
    """
    Get the updated agent heading and speeds from actions

    Args:
        _: Dummy JAX random key.
        params: Agent parameters.
        x: Agent rotation and acceleration actions.

    Returns:
        float: New agent heading.
        float: New agent speed.
    """
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
    _: chex.PRNGKey, _params: None, x: Tuple[chex.Array, float, float]
) -> chex.Array:
    """
    Get updated agent positions from current speed and heading

    Args:
        _: Dummy JAX random key.
        _params: unused parameters.
        x: Tuple containing current agent position, heading and speed.

    Returns:
        jax array (float32): Updated agent position
    """
    pos, heading, speed = x
    d_pos = jnp.array([speed * jnp.cos(heading), speed * jnp.sin(heading)])
    return (pos + d_pos) % 1.0


def init_state(
    n: int, params: types.AgentParams, key: chex.PRNGKey
) -> types.AgentState:
    """
    Randomly initialise state of a group of agents

    Args:
        n: Number of agents to initialise.
        params: Agent parameters.
        key: JAX random key.

    Returns:
        AgentState: Random agent states (i.e. position, headings, and speeds)
    """
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
    """
    Update the state of a group of agents from a sample of actions

    Args:
        key: Dummy JAX random key.
        params: Agent parameters.
        state: Current agent states.
        actions: Agent actions, i.e. a 2D array of action for each agent.

    Returns:
        AgentState: Updated state of the agents after applying steering
            actions and updating positions.
    """
    actions = jax.numpy.clip(actions, min=-1.0, max=1.0)
    headings, speeds = update_velocity(key, params, (actions, state))
    positions = move(key, None, (state.pos, headings, speeds))

    return types.AgentState(
        pos=positions,
        speed=speeds,
        heading=headings,
    )


def view(
    _: chex.PRNGKey,
    params: Tuple[float, float],
    a: types.AgentState,
    b: types.AgentState,
    *,
    n_view: int,
    i_range: float,
) -> chex.Array:
    """
    Simple agent view model

    Simple view model where the agents view angle is subdivided
    into an array of values representing the distance from
    the agent along a rays from the agent, with rays evenly distributed.
    across the agents field of view. The limit of vision is set at 1.0,
    which is also the default value if no object is within range.
    Currently, this model assumes the viewed objects are circular.

    Args:
        _: Dummy JAX random key.
        params: Tuple containing agent view angle and view-radius.
        a: Viewing agent state.
        b: State of agent being viewed.
        n_view: Static number of view rays/subdivisions (i.e. how
            many cells the resulting array contains).
        i_range: Static agent view/interaction range.

    Returns:
        jax array (float32): 1D array representing the distance
            along a ray from the agent to another agent.
    """
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
