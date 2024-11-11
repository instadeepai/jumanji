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


class TargetDynamics(abc.ABC):
    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey, target_pos: chex.Array) -> chex.Array:
        """Interface for target position update function.

        Args:
            key: random key.
            target_pos: Current target positions.

        Returns:
            Updated target positions.
        """


class RandomWalk(TargetDynamics):
    def __init__(self, step_size: float):
        """
        Random walk target dynamics.

        Target positions are updated with random
        steps, sampled uniformly from the range
        [-step-size, step-size].

        Args:
            step_size: Maximum random step-size
        """
        self.step_size = step_size

    def __call__(self, key: chex.PRNGKey, target_pos: chex.Array) -> chex.Array:
        """Update target positions.

        Args:
            key: random key.
            target_pos: Current target positions.

        Returns:
            Updated target positions.
        """
        d_pos = jax.random.uniform(key, target_pos.shape)
        d_pos = self.step_size * 2.0 * (d_pos - 0.5)
        return target_pos + d_pos
