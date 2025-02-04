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

import abc
from typing import Tuple

import chex
import esquilax
import jax.numpy as jnp

from jumanji.environments.swarms.common.types import AgentState
from jumanji.environments.swarms.common.updates import angular_width, view, view_reduction_fn
from jumanji.environments.swarms.search_and_rescue.types import State, TargetState


def view_reduction(view_shape: Tuple[int, ...]) -> esquilax.reductions.Reduction:
    return esquilax.reductions.Reduction(
        fn=view_reduction_fn,
        id=-jnp.ones(view_shape),
    )


class ObservationFn(abc.ABC):
    def __init__(
        self,
        num_channels: int,
        num_vision: int,
        searcher_vision_range: float,
        target_vision_range: float,
        view_angle: float,
        agent_radius: float,
        env_size: float,
    ) -> None:
        """
        Base class for observation function mapping state to individual agent views.

        Maps states to an array of individual local agent views of
        the environment, with shape (n-agents, n-channels, n-vision).
        Channels can be used to differentiate between agent types or
        statuses.

        Args:
            num_channels: Number of channels in agent view.
            num_vision: Size of vision array.
            searcher_vision_range: Range at which other searching agents
                become visible.
            target_vision_range: Range at which targets become visible.
            view_angle: Agent view angle (as a fraction of π).
            agent_radius: Agent/target visual radius.
            env_size: Environment size.
        """
        self.num_channels = num_channels
        self.num_vision = num_vision
        self.searcher_vision_range = searcher_vision_range
        self.target_vision_range = target_vision_range
        self.view_angle = view_angle
        self.agent_radius = agent_radius
        self.env_size = env_size

    @abc.abstractmethod
    def __call__(self, state: State) -> chex.Array:
        """
        Generate agent view/observation from state

        Args:
            state: Current simulation state

        Returns:
            Array of individual agent views (n-agents, n-channels, n-vision).
        """


def _target_view(
    params: Tuple[float, float],
    searcher: AgentState,
    target: TargetState,
    *,
    n_view: int,
    i_range: float,
    env_size: float,
) -> chex.Array:
    """
    Return view of a target, dependent on target status.

    This function is intended to be mapped over agents target
    pairs by Esquilax.

    Args:
        params: View angle and target visual radius.
        searcher: Searcher agent state.
        target: Target state.
        n_view: Number of value sin view array.
        i_range: Vision range.
        env_size: Environment size.

    Returns:
        Segmented agent view of target.
    """
    view_angle, agent_radius = params
    rays = jnp.linspace(
        -view_angle * jnp.pi,
        view_angle * jnp.pi,
        n_view,
        endpoint=True,
    )
    d, left, right = angular_width(
        searcher.pos,
        target.pos,
        searcher.heading,
        i_range,
        agent_radius,
        env_size,
    )
    ray_checks = jnp.logical_and(left < rays, rays < right)
    checks_a = jnp.logical_and(target.found, ray_checks)
    checks_b = jnp.logical_and(~target.found, ray_checks)
    obs = [jnp.where(checks_a, d, -1.0), jnp.where(checks_b, d, -1.0)]
    obs = jnp.vstack(obs)
    return obs


class AgentAndTargetObservationFn(ObservationFn):
    def __init__(
        self,
        num_vision: int,
        searcher_vision_range: float,
        target_vision_range: float,
        view_angle: float,
        agent_radius: float,
        env_size: float,
    ) -> None:
        """
        Vision model that contains other agents, and all targets.

        Searchers and targets are visualised as individual channels,
        with found and unfound targets also shown on different channels.

        Args:
            num_vision: Size of vision array.
            searcher_vision_range: Range at which other searching agents
                become visible.
            target_vision_range: Range at which targets become visible.
            view_angle: Agent view angle (as a fraction of π).
            agent_radius: Agent/target visual radius.
            env_size: Environment size.
        """
        super().__init__(
            3,
            num_vision,
            searcher_vision_range,
            target_vision_range,
            view_angle,
            agent_radius,
            env_size,
        )

    def __call__(self, state: State) -> chex.Array:
        """
        Generate agent view/observation from state

        Args:
            state: Current simulation state

        Returns:
            Array of individual agent views of shape
            (n-agents, 3, n-vision). Other agents are shown
            in channel 0, located targets channel 1, and un-located
            targets in channel 2.
        """
        searcher_views = esquilax.transforms.spatial(
            view,
            reduction=view_reduction((self.num_vision,)),
            include_self=False,
            i_range=self.searcher_vision_range,
            dims=self.env_size,
        )(
            (self.view_angle, self.agent_radius),
            state.searchers,
            state.searchers,
            pos=state.searchers.pos,
            n_view=self.num_vision,
            i_range=self.searcher_vision_range,
            env_size=self.env_size,
        )
        target_views = esquilax.transforms.spatial(
            _target_view,
            reduction=view_reduction((2, self.num_vision)),
            include_self=False,
            i_range=self.target_vision_range,
            dims=self.env_size,
        )(
            (self.view_angle, self.agent_radius),
            state.searchers,
            state.targets,
            pos=state.searchers.pos,
            pos_b=state.targets.pos,
            n_view=self.num_vision,
            i_range=self.target_vision_range,
            env_size=self.env_size,
        )
        return jnp.hstack([searcher_views[:, jnp.newaxis], target_views])
