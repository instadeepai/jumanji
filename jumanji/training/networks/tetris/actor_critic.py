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
import numpy as np

from jumanji.environments.packing.tetris.env import Observation, Tetris
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    FactorisedActionSpaceParametricDistribution,
)


def make_actor_critic_networks_tetris(
    tetris: Tetris,
    conv_num_channels: int,
    tetromino_layers: int,
    final_layer_dims: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `Tetris` environment."""

    parametric_action_distribution = FactorisedActionSpaceParametricDistribution(
        action_spec_num_values=np.asarray(tetris.action_spec().num_values)
    )
    policy_network = make_network_cnn(
        conv_num_channels=conv_num_channels,
        tetromino_layers=tetromino_layers,
        final_layer_dims=final_layer_dims,
        time_limit=tetris.time_limit,
        critic=False,
    )
    value_network = make_network_cnn(
        conv_num_channels=conv_num_channels,
        tetromino_layers=tetromino_layers,
        final_layer_dims=final_layer_dims,
        time_limit=tetris.time_limit,
        critic=True,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_network_cnn(
    conv_num_channels: int,
    tetromino_layers: int,
    final_layer_dims: Sequence[int],
    time_limit: int,
    critic: bool,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        conv_layers = hk.Sequential(
            [
                hk.Conv2D(conv_num_channels, (2, 2), 1, padding="VALID"),
                jax.nn.relu,
                hk.Conv2D(conv_num_channels, (2, 2), 1, padding="VALID"),
                jax.nn.relu,
                hk.Conv2D(conv_num_channels, (2, 2), 1, 2, padding="VALID"),
                jax.nn.relu,
                hk.Flatten(),
            ]
        )
        grid_embeddings = conv_layers(observation.grid.astype(float)[..., None])

        mlp_layers = hk.Sequential(
            [
                hk.Flatten(),
                hk.Linear(tetromino_layers),
                jax.nn.relu,
                hk.Linear(tetromino_layers),
                jax.nn.relu,
            ]
        )

        tetromino_embeddings = mlp_layers(observation.tetromino.astype(float))
        norm_step_count = jnp.expand_dims(observation.step_count / time_limit, axis=-1)

        output = jnp.concatenate(
            [grid_embeddings, tetromino_embeddings, norm_step_count], axis=-1
        )
        final_layers = hk.nets.MLP(final_layer_dims)
        output = final_layers(output)

        if critic:
            return jnp.mean(output, axis=-1)
        else:
            output = output.reshape(-1, 4, 10)
            masked_logits = jnp.where(
                observation.action_mask, output, jnp.finfo(jnp.float32).min
            ).reshape(observation.action_mask.shape[0], -1)
            return masked_logits

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
