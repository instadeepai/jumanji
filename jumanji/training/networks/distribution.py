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
import jax
import jax.numpy as jnp


class Distribution(abc.ABC):
    @abc.abstractmethod
    def sample(self, seed: chex.PRNGKey) -> chex.Array:
        pass

    @abc.abstractmethod
    def mode(self) -> chex.Array:
        pass

    @abc.abstractmethod
    def log_prob(self, x: chex.Array) -> chex.Array:
        pass

    @abc.abstractmethod
    def entropy(self) -> chex.Array:
        pass

    @abc.abstractmethod
    def kl_divergence(self, other: "Distribution") -> chex.Array:
        pass


class CategoricalDistribution(Distribution):
    """Categorical distribution."""

    def __init__(self, logits: chex.Array):
        self.logits = logits
        self.num_actions = jnp.shape(logits)[-1]

    def sample(self, seed: chex.PRNGKey) -> chex.Array:
        return jax.random.categorical(seed, self.logits)

    def mode(self) -> chex.Array:
        return jnp.argmax(self.logits, axis=-1)

    def log_prob(self, x: chex.Array) -> chex.Array:
        value_one_hot = jax.nn.one_hot(x, self.num_actions)
        mask_outside_domain = jnp.logical_or(x < 0, x > self.num_actions - 1)
        safe_log_probs = jnp.where(
            value_one_hot == 0,
            jnp.zeros((), dtype=self.logits.dtype),
            jax.nn.log_softmax(self.logits) * value_one_hot,
        )
        return jnp.where(
            mask_outside_domain,
            -jnp.inf,
            jnp.sum(safe_log_probs, axis=-1),
        )

    def entropy(self) -> chex.Array:
        log_probs = jax.nn.log_softmax(self.logits)
        probs = jax.nn.softmax(self.logits)
        return -jnp.sum(jnp.where(probs == 0, 0.0, probs * log_probs), axis=-1)

    def kl_divergence(  # type: ignore[override]
        self,
        other: "CategoricalDistribution",
    ) -> chex.Array:
        log_probs = jax.nn.log_softmax(self.logits)
        probs = jax.nn.softmax(self.logits)
        log_probs_other = jax.nn.log_softmax(other.logits)
        return jnp.sum(
            jnp.where(probs == 0, 0.0, probs * (log_probs - log_probs_other)), axis=-1
        )
