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
import jax

from jumanji.environments.swarms.search_and_rescue.types import TargetState


class TargetDynamics(abc.ABC):
    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey, targets: TargetState, env_size: float) -> TargetState:
        """Interface for target state update function.

        NOTE: Target positions should be bound to environment
            area (generally wrapped around at the boundaries).

        Args:
            key: Random key.
            targets: Current target states.
            env_size: Environment size.

        Returns:
            Updated target states.
        """


class RandomWalk(TargetDynamics):
    def __init__(self, step_size: float):
        """
        Simple random walk target dynamics.

        Target positions are updated with random steps, sampled uniformly
        from the range `[-step-size, step-size]`.

        Args:
            step_size: Maximum random step-size in each axis.
        """
        self.step_size = step_size

    def __call__(self, key: chex.PRNGKey, targets: TargetState, env_size: float) -> TargetState:
        """Update target state.

        Args:
            key: random key.
            targets: Current target states.
            env_size: Environment size.

        Returns:
            Updated target states.
        """
        d_pos = jax.random.uniform(key, targets.pos.shape)
        d_pos = self.step_size * 2.0 * (d_pos - 0.5)
        pos = (targets.pos + d_pos) % env_size
        return TargetState(pos=pos, vel=targets.vel, found=targets.found)
