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

from typing import Sequence

import chex
import haiku as hk
import jax
import jax.numpy as jnp

from jumanji.environments.routing.sokoban import Observation, Sokoban
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    CategoricalParametricDistribution,
)


def make_actor_critic_networks_sokoban(
    sokoban: Sokoban,
    channels: Sequence[int],
    policy_layers: Sequence[int],
    value_layers: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `Sokoban` environment."""
    num_actions = sokoban.action_spec().num_values
    parametric_action_distribution = CategoricalParametricDistribution(
        num_actions=num_actions
    )

    policy_network = make_sokoban_cnn(
        num_outputs=num_actions,
        mlp_units=policy_layers,
        channels=channels,
        time_limit=sokoban.time_limit,
    )
    value_network = make_sokoban_cnn(
        num_outputs=1,
        mlp_units=value_layers,
        channels=channels,
        time_limit=sokoban.time_limit,
    )

    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_sokoban_cnn(
    num_outputs: int,
    mlp_units: Sequence[int],
    channels: Sequence[int],
    time_limit: int,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:

        # Iterate over the channels sequence to create convolutional layers
        layers = []
        for i, conv_n_channels in enumerate(channels):
            layers.append(hk.Conv2D(conv_n_channels, (3, 3), stride=2 if i == 0 else 1))
            layers.append(jax.nn.relu)

        layers.append(hk.Flatten())

        torso = hk.Sequential(layers)

        x_processed = preprocess_input(observation.grid)

        embedding = torso(x_processed)

        norm_step_count = jnp.expand_dims(observation.step_count / time_limit, axis=-1)
        embedding = jnp.concatenate([embedding, norm_step_count], axis=-1)
        head = hk.nets.MLP((*mlp_units, num_outputs), activate_final=False)
        if num_outputs == 1:
            value = jnp.squeeze(head(embedding), axis=-1)
            return value
        else:
            logits = head(embedding)

            return logits

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def preprocess_input(
    input_array: chex.Array,
) -> chex.Array:

    one_hot_array_fixed = jnp.equal(input_array[..., 0:1], jnp.array([3, 4])).astype(
        jnp.float32
    )

    one_hot_array_variable = jnp.equal(input_array[..., 1:2], jnp.array([1, 2])).astype(
        jnp.float32
    )

    total = jnp.concatenate((one_hot_array_fixed, one_hot_array_variable), axis=-1)

    return total
