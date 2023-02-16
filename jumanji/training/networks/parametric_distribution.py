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
from typing import Any

import chex
import jax.numpy as jnp
import numpy as np

from jumanji.training.networks.distribution import CategoricalDistribution, Distribution
from jumanji.training.networks.postprocessor import IdentityBijector, Postprocessor


class FactorisedActionSpaceReshapeBijector(Postprocessor):
    """Identity bijector that reshapes (flattens and unflattens)
    a sequential action"""

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
        action = jnp.stack(list(reversed(action_components)), axis=-1)
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


class ParametricDistribution(abc.ABC):
    """Abstract class for parametric (action) distribution."""

    def __init__(
        self,
        param_size: int,
        postprocessor: Postprocessor,
        event_ndims: int,
    ):
        """Abstract class for parametric (action) distribution.
        Specifies how to transform distribution parameters (i.e. actor output)
        into a distribution over actions.
        Args:
            param_size: size of the parameters for the distribution
            postprocessor: bijector which is applied after sampling (in practice, it's
                tanh or identity)
            event_ndims: rank of the distribution sample (i.e. action)
        """
        self._param_size = param_size
        self._postprocessor = postprocessor
        self._event_ndims = event_ndims  # rank of events
        assert event_ndims in [0, 1]

    @abc.abstractmethod
    def create_dist(self, parameters: chex.Array) -> Distribution:
        """Creates distribution from parameters."""

    @property
    def param_size(self) -> int:
        return self._param_size

    def postprocess(self, event: chex.Array) -> chex.Array:
        return self._postprocessor.forward(event)

    def inverse_postprocess(self, event: chex.Array) -> chex.Array:
        return self._postprocessor.inverse(event)

    def sample_no_postprocessing(
        self, parameters: chex.Array, seed: chex.PRNGKey
    ) -> Any:
        return self.create_dist(parameters).sample(seed=seed)

    def sample(self, parameters: chex.Array, seed: chex.PRNGKey) -> chex.Array:
        """Returns a sample from the postprocessed distribution."""
        return self.postprocess(self.sample_no_postprocessing(parameters, seed))

    def mode(self, parameters: chex.Array) -> chex.Array:
        """Returns the mode of the postprocessed distribution."""
        return self.postprocess(self.create_dist(parameters).mode())

    def log_prob(self, parameters: chex.Array, raw_actions: chex.Array) -> chex.Array:
        """Compute the log probability of actions."""
        dist = self.create_dist(parameters)
        log_probs = dist.log_prob(raw_actions)
        log_probs -= self._postprocessor.forward_log_det_jacobian(raw_actions)
        if self._event_ndims == 1:
            log_probs = jnp.sum(log_probs, axis=-1)  # sum over action dimension
        else:
            assert self._event_ndims == 0
        return log_probs

    def entropy(self, parameters: chex.Array, seed: chex.PRNGKey) -> chex.Array:
        """Return the entropy of the given distribution."""
        dist = self.create_dist(parameters)
        entropy = dist.entropy()
        entropy += self._postprocessor.forward_log_det_jacobian(dist.sample(seed=seed))
        if self._event_ndims == 1:
            entropy = jnp.sum(entropy, axis=-1)
        else:
            assert self._event_ndims == 0
        return entropy

    def kl_divergence(
        self, parameters: chex.Array, other_parameters: chex.Array
    ) -> chex.Array:
        """KL divergence is invariant with respect to transformation by the same bijector."""
        dist = self.create_dist(parameters)
        other_dist = self.create_dist(other_parameters)
        return dist.kl_divergence(other_dist)


class CategoricalParametricDistribution(ParametricDistribution):
    """Categorical distribution for discrete action spaces."""

    def __init__(self, num_actions: int):
        """Initialize the distribution.
        Args:
            num_actions: the number of actions.
        """
        postprocessor = IdentityBijector()
        super().__init__(
            param_size=num_actions,
            postprocessor=postprocessor,
            event_ndims=0,
        )

    def create_dist(self, parameters: chex.Array) -> CategoricalDistribution:
        return CategoricalDistribution(logits=parameters)


class MultiCategoricalParametricDistribution(ParametricDistribution):
    """Multi categorical distribution for multi discrete action spaces."""

    def __init__(self, num_values: chex.ArrayNumpy):
        """Initialize the distribution.
        Args:
            num_values: ndarray of integers specifying the number of possible values to represent
                for each element of the action vector.
        """
        assert np.ndim(num_values) == 1
        assert np.all(num_values == num_values[0])
        super().__init__(
            param_size=int(np.sum(num_values)),
            postprocessor=IdentityBijector(),
            event_ndims=1,
        )

    def create_dist(self, parameters: chex.Array) -> CategoricalDistribution:
        return CategoricalDistribution(logits=parameters)


class FactorisedActionSpaceParametricDistribution(ParametricDistribution):
    """Categorical distribution for a factorised action space"""

    def __init__(self, action_spec_num_values: chex.ArrayNumpy):
        """Initialize the distribution.
        Args:
            action_spec_num_values: the dimensions of each of the factors in the action space"""
        num_actions = int(np.prod(action_spec_num_values))
        posprocessor = FactorisedActionSpaceReshapeBijector(
            action_spec_num_values=action_spec_num_values
        )
        super().__init__(
            param_size=num_actions, postprocessor=posprocessor, event_ndims=0
        )

    def create_dist(self, parameters: chex.Array) -> CategoricalDistribution:
        return CategoricalDistribution(logits=parameters)
