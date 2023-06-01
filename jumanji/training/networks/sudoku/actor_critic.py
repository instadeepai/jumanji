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

from jumanji.environments.logic.sudoku import Observation, Sudoku
from jumanji.environments.logic.sudoku.constants import BOARD_WIDTH
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    FactorisedActionSpaceParametricDistribution,
)
from jumanji.training.networks.transformer_block import TransformerBlock


def make_cnn_actor_critic_networks_sudoku(
    sudoku: Sudoku,
    num_channels: int,
    policy_layers: Sequence[int],
    value_layers: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `Sudoku` environment. Uses the
    CNN network architecture."""
    num_actions = sudoku.action_spec().num_values
    parametric_action_distribution = FactorisedActionSpaceParametricDistribution(
        action_spec_num_values=np.asarray(num_actions)
    )

    policy_network = make_sudoku_cnn(
        num_outputs=int(np.prod(num_actions)),
        mlp_units=policy_layers,
        conv_n_channels=num_channels,
    )
    value_network = make_sudoku_cnn(
        num_outputs=1,
        mlp_units=value_layers,
        conv_n_channels=num_channels,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_equivariant_actor_critic_networks_sudoku(
    sudoku: Sudoku,
    num_heads: int,
    key_size: int,
    policy_layers: Sequence[int],
    value_layers: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `Sudoku` environment. Uses the
    digits-permutation equivariant network architecture."""
    num_actions = sudoku.action_spec().num_values
    parametric_action_distribution = FactorisedActionSpaceParametricDistribution(
        action_spec_num_values=np.asarray(num_actions)
    )

    policy_network = make_sudoku_equivariant(
        is_critic=False,
        mlp_units=policy_layers,
        key_size=key_size,
        num_heads=num_heads,
    )
    value_network = make_sudoku_equivariant(
        is_critic=True,
        mlp_units=value_layers,
        key_size=key_size,
        num_heads=num_heads,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_sudoku_cnn(
    num_outputs: int,
    mlp_units: Sequence[int],
    conv_n_channels: int,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = hk.Sequential(
            [
                hk.Conv2D(conv_n_channels, (2, 2), 2),
                jax.nn.relu,
                hk.Conv2D(conv_n_channels, (2, 2), 1),
                jax.nn.relu,
                hk.Flatten(),
            ]
        )
        embedding = torso(observation.board[..., None] / BOARD_WIDTH - 0.5)

        head = hk.nets.MLP((*mlp_units, num_outputs), activate_final=False)
        if num_outputs == 1:
            value = jnp.squeeze(head(embedding), axis=-1)
            return value
        else:
            logits = head(embedding)
            logits = logits.reshape(-1, BOARD_WIDTH, BOARD_WIDTH, BOARD_WIDTH)

            logits = jnp.where(
                observation.action_mask, logits, jnp.finfo(jnp.float32).min
            )

            return logits.reshape(observation.action_mask.shape[0], -1)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_sudoku_equivariant(
    is_critic: bool,
    num_heads: int = 4,
    key_size: int = 64,
    mlp_units: Sequence[int] = (64,),
) -> FeedForwardNetwork:
    """A network that is equivariant to a permutation of digits."""

    def network_fn(observation: Observation) -> chex.Array:
        board = observation.board
        board = jax.nn.one_hot(board, BOARD_WIDTH)
        board = board.reshape(board.shape[0], BOARD_WIDTH**2, BOARD_WIDTH)
        board = jnp.transpose(board, (0, 2, 1))
        board = hk.nets.MLP((key_size * num_heads,), activate_final=True)(board)

        embedding = TransformerBlock(
            num_heads=num_heads,
            key_size=key_size,
            mlp_units=mlp_units,
            w_init_scale=1.0,
        )(board, board, board)

        if is_critic:
            logits = hk.Linear(1)(embedding)
            logits = logits.squeeze(axis=-1)
            value = jnp.mean(logits, axis=-1)
            return value
        else:
            logits = hk.Linear(BOARD_WIDTH**2)(embedding)

            logits = jnp.transpose(logits, (0, 2, 1))

            logits = logits.reshape(
                board.shape[0], BOARD_WIDTH, BOARD_WIDTH, BOARD_WIDTH
            )

            logits = jnp.where(
                observation.action_mask, logits, jnp.finfo(jnp.float32).min
            )

            return logits.reshape(observation.action_mask.shape[0], -1)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
