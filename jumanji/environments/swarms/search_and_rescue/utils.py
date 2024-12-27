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
import jax.numpy as jnp
from esquilax.utils import shortest_vector

from jumanji.environments.swarms.common.types import AgentState
from jumanji.environments.swarms.search_and_rescue.types import TargetState


def _check_target_in_view(
    searcher_pos: chex.Array,
    target_pos: chex.Array,
    searcher_heading: chex.Array,
    searcher_view_angle: float,
    env_size: float,
) -> chex.Array:
    """
    Check if a target is inside the view-cone of a searcher.

    Args:
        searcher_pos: Searcher position
        target_pos: Target position
        searcher_heading: Searcher heading angle
        searcher_view_angle: Searcher view angle
        env_size: Size of the environment

    Returns:
        bool: Flag indicating if a target is within view.
    """
    dx = shortest_vector(searcher_pos, target_pos, length=env_size)
    phi = jnp.arctan2(dx[1], dx[0]) % (2 * jnp.pi)
    dh = shortest_vector(phi, searcher_heading, 2 * jnp.pi)
    searcher_view_angle = searcher_view_angle * jnp.pi
    return (dh >= -searcher_view_angle) & (dh <= searcher_view_angle)


def searcher_detect_targets(
    searcher_view_angle: float,
    searcher: AgentState,
    target: Tuple[chex.Array, TargetState],
    *,
    env_size: float,
    n_targets: int,
) -> chex.Array:
    """
    Return array of flags indicating if a target has been located

    Sets the flag at the target index if the target is within the
    searchers view cone, and has not already been detected.

    Args:
        searcher_view_angle: View angle of searching agents
            representing a fraction of pi from the agents heading.
        searcher: State of the searching agent (i.e. the agent
            position and heading)
        target: Index and State of the target (i.e. its position and
            search status).
        env_size: size of the environment.
        n_targets: Number of search targets (static).

    Returns:
        array of boolean flags, set if a target at the index has been found.
    """
    target_idx, target = target
    target_found = jnp.zeros((n_targets,), dtype=bool)
    can_see = _check_target_in_view(
        searcher.pos, target.pos, searcher.heading, searcher_view_angle, env_size
    )
    return target_found.at[target_idx].set(jnp.logical_and(~target.found, can_see))
