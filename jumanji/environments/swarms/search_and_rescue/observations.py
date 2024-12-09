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
import jax.numpy as jnp
from esquilax.transforms import spatial

from jumanji.environments.swarms.common.types import AgentState
from jumanji.environments.swarms.common.updates import angular_width, view, view_reduction
from jumanji.environments.swarms.search_and_rescue.types import State, TargetState


class ObservationFn(abc.ABC):
    def __init__(
        self,
        view_shape: Tuple[int, ...],
        num_vision: int,
        vision_range: float,
        view_angle: float,
        agent_radius: float,
        env_size: float,
    ) -> None:
        """
        Base class for observation function mapping state to individual agent views.

        Args:
            view_shape: Individual agent view shape.
            num_vision: Size of vision array.
            vision_range: Vision range.
            view_angle: Agent view angle (as a fraction of pi).
            agent_radius: Agent/target visual radius.
            env_size: Environment size.
        """
        self.view_shape = view_shape
        self.num_vision = num_vision
        self.vision_range = vision_range
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


class AgentObservationFn(ObservationFn):
    def __init__(
        self,
        num_vision: int,
        vision_range: float,
        view_angle: float,
        agent_radius: float,
        env_size: float,
    ) -> None:
        """
        Observation that only visualises other search agents in proximity.

        Args:
            num_vision: Size of vision array.
            vision_range: Vision range.
            view_angle: Agent view angle (as a fraction of pi).
            agent_radius: Agent/target visual radius.
            env_size: Environment size.
        """
        super().__init__(
            (1, num_vision),
            num_vision,
            vision_range,
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
            (n-agents, 1, n-vision).
        """
        searcher_views = spatial(
            view,
            reduction=view_reduction,
            default=-jnp.ones((self.num_vision,)),
            include_self=False,
            i_range=self.vision_range,
            dims=self.env_size,
        )(
            state.key,
            (self.view_angle, self.agent_radius),
            state.searchers,
            state.searchers,
            pos=state.searchers.pos,
            n_view=self.num_vision,
            i_range=self.vision_range,
            env_size=self.env_size,
        )
        return searcher_views[:, jnp.newaxis]


def found_target_view(
    _key: chex.PRNGKey,
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

    This function is intended to be mapped over agents-targets
    by Esquilax.

    Args:
        _key: Dummy random key (required by Esquilax).
        params: View angle and target visual radius.
        searcher: Searcher agent state
        target: Target state
        n_view: Number of value sin view array.
        i_range: Vision range
        env_size: Environment size

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
    checks = jnp.logical_and(target.found, jnp.logical_and(left < rays, rays < right))
    obs = jnp.where(checks, d, -1.0)
    return obs


class AgentAndTargetObservationFn(ObservationFn):
    def __init__(
        self,
        num_vision: int,
        vision_range: float,
        view_angle: float,
        agent_radius: float,
        env_size: float,
    ) -> None:
        """
        Vision model that contains other agents, and found targets.

        Searchers and targets are visualised as individual channels.
        Targets are only included if they have been located already.

        Args:
            num_vision: Size of vision array.
            vision_range: Vision range.
            view_angle: Agent view angle (as a fraction of pi).
            agent_radius: Agent/target visual radius.
            env_size: Environment size.
        """
        self.vision_range = vision_range
        self.view_angle = view_angle
        self.agent_radius = agent_radius
        self.env_size = env_size
        super().__init__(
            (2, num_vision),
            num_vision,
            vision_range,
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
            Array of individual agent views
        """
        searcher_views = spatial(
            view,
            reduction=view_reduction,
            default=-jnp.ones((self.num_vision,)),
            include_self=False,
            i_range=self.vision_range,
            dims=self.env_size,
        )(
            state.key,
            (self.view_angle, self.agent_radius),
            state.searchers,
            state.searchers,
            pos=state.searchers.pos,
            n_view=self.num_vision,
            i_range=self.vision_range,
            env_size=self.env_size,
        )
        target_views = spatial(
            found_target_view,
            reduction=view_reduction,
            default=-jnp.ones((self.num_vision,)),
            include_self=False,
            i_range=self.vision_range,
            dims=self.env_size,
        )(
            state.key,
            (self.view_angle, self.agent_radius),
            state.searchers,
            state.targets,
            pos=state.searchers.pos,
            pos_b=state.targets.pos,
            n_view=self.num_vision,
            i_range=self.vision_range,
            env_size=self.env_size,
        )
        return jnp.hstack([searcher_views[:, jnp.newaxis], target_views[:, jnp.newaxis]])


def all_target_view(
    _key: chex.PRNGKey,
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

    This function is intended to be mapped over agents-targets
    by Esquilax.

    Args:
        _key: Dummy random key (required by Esquilax).
        params: View angle and target visual radius.
        searcher: Searcher agent state
        target: Target state
        n_view: Number of value sin view array.
        i_range: Vision range
        env_size: Environment size

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


class AgentAndAllTargetObservationFn(ObservationFn):
    def __init__(
        self,
        num_vision: int,
        vision_range: float,
        view_angle: float,
        agent_radius: float,
        env_size: float,
    ) -> None:
        """
        Vision model that contains other agents, and found targets.

        Searchers and targets are visualised as individual channels.
        Targets are only included if they have been located already.

        Args:
            num_vision: Size of vision array.
            vision_range: Vision range.
            view_angle: Agent view angle (as a fraction of pi).
            agent_radius: Agent/target visual radius.
            env_size: Environment size.
        """
        self.vision_range = vision_range
        self.view_angle = view_angle
        self.agent_radius = agent_radius
        self.env_size = env_size
        super().__init__(
            (3, num_vision),
            num_vision,
            vision_range,
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
            Array of individual agent views
        """
        searcher_views = spatial(
            view,
            reduction=view_reduction,
            default=-jnp.ones((self.num_vision,)),
            include_self=False,
            i_range=self.vision_range,
            dims=self.env_size,
        )(
            state.key,
            (self.view_angle, self.agent_radius),
            state.searchers,
            state.searchers,
            pos=state.searchers.pos,
            n_view=self.num_vision,
            i_range=self.vision_range,
            env_size=self.env_size,
        )
        target_views = spatial(
            all_target_view,
            reduction=view_reduction,
            default=-jnp.ones((2, self.num_vision)),
            include_self=False,
            i_range=self.vision_range,
            dims=self.env_size,
        )(
            state.key,
            (self.view_angle, self.agent_radius),
            state.searchers,
            state.targets,
            pos=state.searchers.pos,
            pos_b=state.targets.pos,
            n_view=self.num_vision,
            i_range=self.vision_range,
            env_size=self.env_size,
        )
        return jnp.hstack([searcher_views[:, jnp.newaxis], target_views])
