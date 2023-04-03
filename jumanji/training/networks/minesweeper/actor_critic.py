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

from jumanji.environments.logic.minesweeper.constants import PATCH_SIZE
from jumanji.environments.logic.minesweeper.env import Minesweeper, Observation
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    FactorisedActionSpaceParametricDistribution,
)


def make_actor_critic_networks_minesweeper(
    minesweeper: Minesweeper,
    board_embed_dim: int,
    board_conv_channels: Sequence[int],
    board_kernel_shape: int,
    num_mines_embed_dim: int,
    final_layer_dims: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `Minesweeper` environment."""
    board_num_rows = minesweeper.num_rows
    board_num_cols = minesweeper.num_cols
    vocab_size = 1 + PATCH_SIZE**2  # unexplored, or 0, 1, ..., 8

    parametric_action_distribution = FactorisedActionSpaceParametricDistribution(
        action_spec_num_values=np.asarray(minesweeper.action_spec().num_values)
    )
    policy_network = make_network_cnn(
        vocab_size=vocab_size,
        board_num_rows=board_num_rows,
        board_num_cols=board_num_cols,
        board_embed_dim=board_embed_dim,
        board_conv_channels=board_conv_channels,
        board_kernel_shape=board_kernel_shape,
        num_mines_embed_dim=num_mines_embed_dim,
        final_layer_dims=final_layer_dims,
        critic=False,
    )
    value_network = make_network_cnn(
        vocab_size=vocab_size,
        board_num_rows=board_num_rows,
        board_num_cols=board_num_cols,
        board_embed_dim=board_embed_dim,
        board_conv_channels=board_conv_channels,
        board_kernel_shape=board_kernel_shape,
        num_mines_embed_dim=num_mines_embed_dim,
        final_layer_dims=final_layer_dims,
        critic=True,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_network_cnn(
    vocab_size: int,
    board_num_rows: int,
    board_num_cols: int,
    board_embed_dim: int,
    board_conv_channels: Sequence[int],
    board_kernel_shape: int,
    num_mines_embed_dim: int,
    final_layer_dims: Sequence[int],
    critic: bool,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        conv_layers = [
            [
                hk.Conv2D(
                    output_channels=output_channels, kernel_shape=board_kernel_shape
                ),
                jax.nn.relu,
            ]
            for output_channels in board_conv_channels
        ]
        board_embedder = hk.Sequential(
            [
                hk.Embed(vocab_size=vocab_size, embed_dim=board_embed_dim),
                *[layer for conv_layer in conv_layers for layer in conv_layer],
            ]
        )
        x = board_embedder(observation.board + 1)
        num_mines_embedder = hk.Linear(num_mines_embed_dim)
        y = num_mines_embedder(
            observation.num_mines[:, None] / (board_num_rows * board_num_cols)
        )[:, None, None, :]
        y = jnp.tile(y, [1, board_num_rows, board_num_cols, 1])
        output = jnp.concatenate([x, y], axis=-1)
        final_layers = hk.nets.MLP((*final_layer_dims, 1))
        output = jnp.squeeze(final_layers(output), axis=-1)
        if critic:
            return jnp.mean(output, axis=(-1, -2))
        else:
            masked_logits = jnp.where(
                observation.action_mask, output, jnp.finfo(jnp.float32).min
            ).reshape(observation.action_mask.shape[0], -1)
            return masked_logits

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
