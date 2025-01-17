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
import jax.numpy as jnp

from jumanji.environments.swarms.search_and_rescue.types import TargetState


class TargetDynamics(abc.ABC):
    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey, targets: TargetState, env_size: float) -> TargetState:
        """Interface for target state update function.

        NOTE: Target positions should be inside the bounds
            of the environment. Out-of-bound co-ordinates can
            lead to unexpected behaviour.

        Args:
            key: Random key.
            targets: Current target states.
            env_size: Environment size.

        Returns:
            Updated target states.
        """


class RandomWalk(TargetDynamics):
    def __init__(self, acc_std: float, vel_max: float):
        """
        Simple random walk target dynamics.

        Target velocities are updated with values sampled from
        a normal distribution with width `acc_std`, and the
        magnitude of the updated velocity clipped to `vel_max`.

        Args:
            acc_std: Standard deviation of acceleration distribution
            vel_max: Max velocity magnitude.
        """
        self.acc_std = acc_std
        self.vel_max = vel_max

    def __call__(self, key: chex.PRNGKey, targets: TargetState, env_size: float) -> TargetState:
        """Update target state.

        Args:
            key: Random key.
            targets: Current target states.
            env_size: Environment size.

        Returns:
            Updated target states.
        """
        acc = self.acc_std * jax.random.normal(key, targets.pos.shape)
        vel = targets.vel + acc
        norm = jnp.sqrt(jnp.sum(vel * vel, axis=1))
        vel = jnp.where(
            norm[:, jnp.newaxis] > self.vel_max, self.vel_max * vel / norm[:, jnp.newaxis], vel
        )
        pos = (targets.pos + vel) % env_size
        return targets.replace(pos=pos, vel=vel)  # type: ignore
