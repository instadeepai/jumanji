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
