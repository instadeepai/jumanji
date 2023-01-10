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

from typing import Sequence, Union

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from jumanji.environments.routing.routing import Routing
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    MultiCategoricalParametricDistribution,
)


def make_actor_critic_networks_routing(
    routing: Routing,
    num_channels: int,
    policy_layers: Sequence[int],
    value_layers: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for Routing."""
    num_values = np.asarray(routing.action_spec().num_values)
    parametric_action_distribution = MultiCategoricalParametricDistribution(
        num_values=num_values
    )
    policy_network = make_network_routing(
        critic=False,
        num_values=num_values,
        mlp_units=policy_layers,
        conv_n_channels=num_channels,
    )
    value_network = make_network_routing(
        critic=True,
        num_values=None,
        mlp_units=value_layers,
        conv_n_channels=num_channels,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_network_routing(
    critic: bool,
    num_values: Union[None, np.ndarray],
    mlp_units: Sequence[int],
    conv_n_channels: int,
) -> FeedForwardNetwork:
    def network_fn(
        observation: chex.Array,
    ) -> chex.Array:
        observation = jnp.asarray(observation[..., 0:, :, :], float)
        torso = hk.Sequential(
            [
                hk.Conv2D(conv_n_channels, (2, 2), 2),
                jax.nn.relu,
                hk.Conv2D(conv_n_channels, (2, 2), 2),
                jax.nn.relu,
                hk.Flatten(),
            ]
        )
        if observation.ndim == 5:
            torso = jax.vmap(torso)
        x = torso(observation)
        if critic:
            head = hk.nets.MLP((*mlp_units, 1), activate_final=False)
            return jnp.squeeze(head(x), axis=-1)
        else:
            assert num_values is not None
            heads = [
                hk.nets.MLP((*mlp_units, num_actions), activate_final=False)
                for num_actions in num_values
            ]
            return jnp.stack([head(x) for head in heads], axis=-2)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
