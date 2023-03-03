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

import chex
from jax import numpy as jnp

from jumanji import specs
from jumanji.environments.logic.planar import utils
from jumanji.environments.logic.planar.types import State


class RewardFn(abc.ABC):
    @abc.abstractmethod
    def __call__(self, state: State, prev_state: State) -> chex.Numeric:
        """Compute the reward based on the current and previous states."""

    @abc.abstractmethod
    def spec(self) -> specs.Array:
        """The spec defining the `dtype` and `shape` of the reward."""


class IntersectionCountRewardFn(RewardFn):
    """This reward returns minus the number of intersections/crossings at each step."""

    def spec(self) -> specs.Array:
        return specs.Array(shape=(), dtype=float, name="reward")

    def __call__(self, state: State, prev_state: State) -> chex.Numeric:
        del prev_state

        segments = jnp.take(state.nodes, state.edges, axis=0)
        num_intersections = utils.intersection_count(segments, state.edges)
        reward = -jnp.array(num_intersections, float)

        return reward


class IntersectionCountChangeRewardFn(RewardFn):
    """This reward returns the change in the number of intersections
    as a consequence of the latest action.

    For example, if an action yielded (resolved) 2 crossings, then the reward is -2 (+2).
    If the number of crossings remains unchanged, the reward is zero.
    """

    def spec(self) -> specs.Array:
        return specs.Array(shape=(), dtype=float, name="reward")

    def __call__(self, state: State, prev_state: State) -> chex.Numeric:
        # Previous number of intersections
        segments = jnp.take(prev_state.nodes, prev_state.edges, axis=0)
        prev_num_intersections = utils.intersection_count(segments, prev_state.edges)

        # New number of intersections
        segments = jnp.take(state.nodes, state.edges, axis=0)
        curr_num_intersections = utils.intersection_count(segments, state.edges)

        # Cast to float
        reward = jnp.float_(prev_num_intersections - curr_num_intersections)

        return reward
