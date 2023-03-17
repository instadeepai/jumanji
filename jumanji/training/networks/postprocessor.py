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

"""Adapted from Brax."""

import abc

import chex
import jax.numpy as jnp


class Postprocessor(abc.ABC):
    @abc.abstractmethod
    def forward(self, x: chex.Array) -> chex.Array:
        pass

    @abc.abstractmethod
    def inverse(self, y: chex.Array) -> chex.Array:
        pass

    @abc.abstractmethod
    def forward_log_det_jacobian(self, x: chex.Array) -> chex.Array:
        pass


class IdentityBijector(Postprocessor):
    """Identity Bijector."""

    def forward(self, x: chex.Array) -> chex.Array:
        return x

    def inverse(self, y: chex.Array) -> chex.Array:
        return y

    def forward_log_det_jacobian(self, x: chex.Array) -> chex.Array:
        return jnp.zeros_like(x, x.dtype)


class FactorisedActionSpaceReshapeBijector(Postprocessor):
    """Identity bijector that reshapes (flattens and unflattens) a sequential action."""

    def __init__(self, action_spec_num_values: chex.ArrayNumpy):
        super().__init__()
        self.action_spec_num_values = action_spec_num_values

    def forward(self, x: chex.Array) -> chex.Array:
        action_components = []
        flat_action = x
        n = self.action_spec_num_values.shape[0]
        for i in range(n - 1, 0, -1):
            flat_action, remainder = jnp.divmod(
                flat_action, self.action_spec_num_values[i]
            )
            action_components.append(remainder)
        action_components.append(flat_action)
        action = jnp.stack(
            list(reversed(action_components)),
            axis=-1,
            dtype=self.action_spec_num_values.dtype,
        )
        return action

    def inverse(self, y: chex.Array) -> chex.Array:
        n = self.action_spec_num_values.shape[0]
        action_components = jnp.split(y, n, axis=-1)
        flat_action = action_components[0]
        for i in range(1, n):
            flat_action = (
                self.action_spec_num_values[i] * flat_action + action_components[i]
            )
        return flat_action

    def forward_log_det_jacobian(self, x: chex.Array) -> chex.Array:
        return jnp.zeros_like(x, x.dtype)
