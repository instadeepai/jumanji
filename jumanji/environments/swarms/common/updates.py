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

from jumanji.environments.swarms.common import types


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


def move(pos: chex.Array, heading: chex.Array, speed: chex.Array, env_size: float) -> chex.Array:
    """
    Get updated agent positions from current speed and heading

    Args:
        pos: Agent position.
        heading: Agent heading (angle).
        speed: Agent speed.
        env_size: Size of the environment.

    Returns:
        jax array (float32): Updated agent position.
    """
    d_pos = jnp.array([speed * jnp.cos(heading), speed * jnp.sin(heading)])
    return (pos + d_pos) % env_size


def update_state(
    key: chex.PRNGKey,
    env_size: float,
    params: types.AgentParams,
    state: types.AgentState,
    actions: chex.Array,
) -> types.AgentState:
    """
    Update the state of a group of agents from a sample of actions

    Args:
        key: Dummy JAX random key.
        env_size: Size of the environment.
        params: Agent parameters.
        state: Current agent states.
        actions: Agent actions, i.e. a 2D array of action for each agent.

    Returns:
        AgentState: Updated state of the agents after applying steering
            actions and updating positions.
    """
    actions = jnp.clip(actions, min=-1.0, max=1.0)
    headings, speeds = update_velocity(key, params, (actions, state))
    positions = jax.vmap(move, in_axes=(0, 0, 0, None))(state.pos, headings, speeds, env_size)

    return types.AgentState(
        pos=positions,
        speed=speeds,
        heading=headings,
    )


def view_reduction(view_a: chex.Array, view_b: chex.Array) -> chex.Array:
    """
    Binary view reduction function.

    Handles reduction where a value of -1.0 indicates no
    agent in view-range. Returns the min value of they
    are both positive, but the max value if one or both of
    the values is -1.0.

    Args:
        view_a: View vector.
        view_b: View vector.

    Returns:
        jax array (float32): View vector indicating the
            shortest distance to the nearest neighbour or
            -1.0 if no agent is present along a ray.
    """
    return jnp.where(
        jnp.logical_or(view_a < 0.0, view_b < 0.0),
        jnp.maximum(view_a, view_b),
        jnp.minimum(view_a, view_b),
    )


def angular_width(
    viewing_pos: chex.Array,
    viewed_pos: chex.Array,
    viewing_heading: chex.Array,
    i_range: float,
    agent_radius: float,
    env_size: float,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """
    Get the normalised distance, and left and right angles to another agent.

    Args:
        viewing_pos: Co-ordinates of the viewing agent
        viewed_pos: Co-ordinates of the viewed agent
        viewing_heading: Heading of viewing agent
        i_range: Interaction range
        agent_radius: Agent visual radius
        env_size: Environment size

    Returns:
        Normalised distance between agents, and the left and right
        angles to the edges of the agent.
    """
    dx = esquilax.utils.shortest_vector(viewing_pos, viewed_pos, length=env_size)
    dist = jnp.sqrt(jnp.sum(dx * dx))
    phi = jnp.arctan2(dx[1], dx[0]) % (2 * jnp.pi)
    dh = esquilax.utils.shortest_vector(phi, viewing_heading, 2 * jnp.pi)
    a_width = jnp.arctan2(agent_radius, dist)
    norm_dist = dist / i_range
    return norm_dist, dh - a_width, dh + a_width


def view(
    _key: chex.PRNGKey,
    params: Tuple[float, float],
    viewing_agent: types.AgentState,
    viewed_agent: types.AgentState,
    *,
    n_view: int,
    i_range: float,
    env_size: float,
) -> chex.Array:
    """
    Simple agent view model

    Simple view model where the agents view angle is subdivided
    into an array of values representing the distance from
    the agent along a rays from the agent, with rays evenly distributed.
    across the agents field of view. The limit of vision is set at 1.0.
    The default value if no object is within range is -1.0.
    Currently, this model assumes the viewed objects are circular.

    Args:
        _key: Dummy JAX random key, required by esquilax API, but
            not used during the interaction.
        params: Tuple containing agent view angle and view-radius.
        viewing_agent: Viewing agent state.
        viewed_agent: State of agent being viewed.
        n_view: Static number of view rays/subdivisions (i.e. how
            many cells the resulting array contains).
        i_range: Static agent view/interaction range.
        env_size: Size of the environment.

    Returns:
        jax array (float32): 1D array representing the distance
            along a ray from the agent to another agent.
    """
    view_angle, agent_radius = params
    rays = jnp.linspace(
        -view_angle * jnp.pi,
        view_angle * jnp.pi,
        n_view,
        endpoint=True,
    )
    d, left, right = angular_width(
        viewing_agent.pos,
        viewed_agent.pos,
        viewing_agent.heading,
        i_range,
        agent_radius,
        env_size,
    )
    obs = jnp.where(jnp.logical_and(left < rays, rays < right), d, -1.0)
    return obs
