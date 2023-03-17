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

from jumanji.environments.logic.game_2048 import Game2048, Observation
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    CategoricalParametricDistribution,
)


def make_actor_critic_networks_game_2048(
    game_2048: Game2048,
    num_channels: int,
    policy_layers: Sequence[int],
    value_layers: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `Game2048` environment."""
    num_actions = game_2048.action_spec().num_values
    parametric_action_distribution = CategoricalParametricDistribution(
        num_actions=num_actions
    )
    policy_network = make_network_cnn(
        num_outputs=num_actions,
        mlp_units=policy_layers,
        conv_n_channels=num_channels,
    )
    value_network = make_network_cnn(
        num_outputs=1,
        mlp_units=value_layers,
        conv_n_channels=num_channels,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_network_cnn(
    num_outputs: int,
    mlp_units: Sequence[int],
    conv_n_channels: int,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        board = observation.board.astype(float)[..., None]
        torso = hk.Sequential(
            [
                hk.Conv2D(conv_n_channels, (2, 2), 1, padding="VALID"),
                jax.nn.relu,
                hk.Conv2D(conv_n_channels, (2, 2), 1, padding="VALID"),
                jax.nn.relu,
                hk.Conv2D(conv_n_channels, (2, 2), 1),
                jax.nn.relu,
                hk.Flatten(),
            ]
        )
        embedding = torso(board)
        head = hk.nets.MLP((*mlp_units, num_outputs), activate_final=False)
        if num_outputs == 1:
            return jnp.squeeze(head(embedding), axis=-1)
        else:
            logits = head(embedding)
            masked_logits = jnp.where(
                observation.action_mask, logits, jnp.finfo(jnp.float32).min
            )
            return masked_logits

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
