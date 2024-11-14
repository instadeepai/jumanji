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
import jax
import jax.numpy as jnp
from esquilax.utils import shortest_vector

from jumanji.environments.swarms.common.types import AgentState
from jumanji.environments.swarms.search_and_rescue.types import TargetState


def has_been_found(
    _key: chex.PRNGKey,
    searcher_view_angle: float,
    target_pos: chex.Array,
    searcher: AgentState,
) -> chex.Array:
    """
    Returns True a target has been found.

    Return true if a target is within detection range
    and within the view cone of a searcher. Used
    to mark targets as found.

    Args:
        _key: Dummy random key (required by Esquilax).
        searcher_view_angle: View angle of searching agents
            representing a fraction of pi from the agents heading.
        target_pos: jax array (float) if shape (2,) representing
            the position of the target.
        searcher: Searcher agent state (i.e. position and heading).

    Returns:
        is-found: `bool` True if the target had been found/detected.
    """
    dx = shortest_vector(searcher.pos, target_pos)
    phi = jnp.arctan2(dx[1], dx[0]) % (2 * jnp.pi)
    dh = shortest_vector(phi, searcher.heading, 2 * jnp.pi)
    searcher_view_angle = searcher_view_angle * jnp.pi
    return (dh >= -searcher_view_angle) & (dh <= searcher_view_angle)


def has_found_target(
    _key: chex.PRNGKey,
    searcher_view_angle: float,
    searcher: AgentState,
    target: TargetState,
) -> chex.Array:
    """
    Return +1.0 reward if the agent has detected an agent.

    Generate rewards for agents if a target is inside the
    searchers view cone, and had not already been detected.

    Args:
        _key: Dummy random key (required by Esquilax).
        searcher_view_angle: View angle of searching agents
            representing a fraction of pi from the agents heading.
        searcher: State of the searching agent (i.e. the agent
            position and heading)
        target: State of the target (i.e. its position and
            search status).

    Returns:
        reward: +1.0 reward if the agent detects a new target.
    """
    dx = shortest_vector(searcher.pos, target.pos)
    phi = jnp.arctan2(dx[1], dx[0]) % (2 * jnp.pi)
    dh = shortest_vector(phi, searcher.heading, 2 * jnp.pi)
    searcher_view_angle = searcher_view_angle * jnp.pi
    can_see = (dh >= -searcher_view_angle) & (dh <= searcher_view_angle)
    return jax.lax.cond(
        ~target.found & can_see,
        lambda: 1.0,
        lambda: 0.0,
    )
