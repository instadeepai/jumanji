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
    tetromino_layers: Sequence[int],
    head_layers: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `Tetris` environment."""

    parametric_action_distribution = FactorisedActionSpaceParametricDistribution(
        action_spec_num_values=np.asarray(tetris.action_spec().num_values)
    )
    policy_network = make_network_cnn(
        conv_num_channels=conv_num_channels,
        tetromino_layers=tetromino_layers,
        head_layers=head_layers,
        time_limit=tetris.time_limit,
        critic=False,
    )
    value_network = make_network_cnn(
        conv_num_channels=conv_num_channels,
        tetromino_layers=tetromino_layers,
        head_layers=head_layers,
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
    tetromino_layers: Sequence[int],
    head_layers: Sequence[int],
    time_limit: int,
    critic: bool,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        grid_net = hk.Sequential(
            [
                hk.Conv2D(conv_num_channels, (3, 5), (1, 1)),
                jax.nn.relu,
                hk.Conv2D(conv_num_channels, (3, 5), (2, 1)),
                jax.nn.relu,
                hk.Conv2D(conv_num_channels, (3, 5), (2, 1)),
                jax.nn.relu,
                hk.Conv2D(conv_num_channels, (3, 3), (2, 1)),
                jax.nn.relu,
            ]
        )
        grid_embeddings = grid_net(
            observation.grid.astype(float)[..., None]
        )  # [B, 2, 10, 64]
        grid_embeddings = jnp.transpose(grid_embeddings, [0, 2, 1, 3])  # [B, 10, 2, 64]
        grid_embeddings = jnp.reshape(
            grid_embeddings, [*grid_embeddings.shape[:2], -1]
        )  # [B, 10, 128]

        tetromino_net = hk.Sequential(
            [
                hk.Flatten(),
                hk.nets.MLP(tetromino_layers, activate_final=True),
            ]
        )
        tetromino_embeddings = tetromino_net(observation.tetromino.astype(float))
        tetromino_embeddings = jnp.tile(
            tetromino_embeddings[:, None], (grid_embeddings.shape[1], 1)
        )
        norm_step_count = observation.step_count / time_limit
        norm_step_count = jnp.tile(
            norm_step_count[:, None, None], (grid_embeddings.shape[1], 1)
        )

        embedding = jnp.concatenate(
            [grid_embeddings, tetromino_embeddings, norm_step_count], axis=-1
        )  # [B, 10, 145]

        if critic:
            embedding = jnp.sum(embedding, axis=-2)  # [B, 145]
            value = hk.nets.MLP((*head_layers, 1))(embedding)  # [B, 1]
            return jnp.squeeze(value, axis=-1)  # [B]
        else:
            num_rotations = observation.action_mask.shape[-2]
            logits = hk.nets.MLP((*head_layers, num_rotations))(embedding)  # [B, 10, 4]
            logits = jnp.transpose(logits, [0, 2, 1])  # [B, 4, 10]
            masked_logits = jnp.where(
                observation.action_mask, logits, jnp.finfo(jnp.float32).min
            ).reshape(observation.action_mask.shape[0], -1)
            return masked_logits  # [B, 40]

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
