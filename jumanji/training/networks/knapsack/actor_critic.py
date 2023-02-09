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

import chex
import haiku as hk
import jax.numpy as jnp

from jumanji.environments.packing.knapsack import Knapsack, Observation
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.encoder_decoder import (
    CriticDecoderBase,
    EncoderBase,
    PolicyDecoderBase,
)
from jumanji.training.networks.parametric_distribution import (
    CategoricalParametricDistribution,
)


def make_actor_critic_networks_knapsack(
    knapsack: Knapsack,
    encoder_num_layers: int,
    encoder_num_heads: int,
    encoder_key_size: int,
    encoder_model_size: int,
    encoder_expand_factor: int,
    decoder_num_heads: int,
    decoder_key_size: int,
    decoder_model_size: int,
) -> ActorCriticNetworks:
    """Make actor-critic networks for Knapsack."""
    num_actions = knapsack.action_spec().num_values
    parametric_action_distribution = CategoricalParametricDistribution(
        num_actions=num_actions
    )
    policy_network = make_network_knapsack(
        num_outputs=num_actions,
        encoder_num_layers=encoder_num_layers,
        encoder_num_heads=encoder_num_heads,
        encoder_key_size=encoder_key_size,
        encoder_model_size=encoder_model_size,
        encoder_expand_factor=encoder_expand_factor,
        decoder_num_heads=decoder_num_heads,
        decoder_key_size=decoder_key_size,
        decoder_model_size=decoder_model_size,
    )
    value_network = make_network_knapsack(
        num_outputs=1,
        encoder_num_layers=encoder_num_layers,
        encoder_num_heads=encoder_num_heads,
        encoder_key_size=encoder_key_size,
        encoder_model_size=encoder_model_size,
        encoder_expand_factor=encoder_expand_factor,
        decoder_num_heads=decoder_num_heads,
        decoder_key_size=decoder_key_size,
        decoder_model_size=decoder_model_size,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_network_knapsack(
    num_outputs: int,
    encoder_num_layers: int,
    encoder_num_heads: int,
    encoder_key_size: int,
    encoder_model_size: int,
    encoder_expand_factor: int,
    decoder_num_heads: int,
    decoder_key_size: int,
    decoder_model_size: int,
) -> FeedForwardNetwork:
    def network_fn(
        observation: Observation,
    ) -> chex.Array:
        encoder = Encoder(
            num_layers=encoder_num_layers,
            num_heads=encoder_num_heads,
            key_size=encoder_key_size,
            model_size=encoder_model_size,
            expand_factor=encoder_expand_factor,
        )
        problem = jnp.concatenate(
            (observation.weights.reshape(-1, 1), observation.values.reshape(-1, 1)),
            axis=-1,
        )
        embedding = encoder(problem)
        if num_outputs == 1:
            decoder = CriticDecoder(
                num_heads=decoder_num_heads,
                key_size=decoder_key_size,
                model_size=decoder_model_size,
            )
        else:
            decoder = PolicyDecoder(
                num_heads=decoder_num_heads,
                key_size=decoder_key_size,
                model_size=decoder_model_size,
            )
        return decoder(observation, embedding)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


class Encoder(EncoderBase):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        key_size: int,
        model_size: int,
        expand_factor: int,
    ):
        super().__init__(num_layers, num_heads, key_size, model_size, expand_factor)

    def get_problem_projection(self, problem: chex.Array) -> chex.Array:
        proj = hk.Linear(self.model_size, name="encoder")
        return proj(problem)


def get_context(observation: Observation, embeddings: chex.Array) -> chex.Array:
    all_items_embedding = jnp.mean(embeddings, axis=-2)
    possible_items_embedding = jnp.mean(
        embeddings, axis=-2, where=observation.action_mask[:, :, None]
    )
    return jnp.concatenate(
        [
            all_items_embedding,
            possible_items_embedding,
        ],
        axis=-1,
    )[:, None, :]


class PolicyDecoder(PolicyDecoderBase):
    def __init__(self, num_heads: int, key_size: int, model_size: int):
        super().__init__(num_heads, key_size, model_size)

    def get_context(  # type: ignore[override]
        self, observation: Observation, embeddings: chex.Array
    ) -> chex.Array:
        return get_context(observation, embeddings)

    def get_transformed_attention_mask(self, attention_mask: chex.Array) -> chex.Array:
        return attention_mask


class CriticDecoder(CriticDecoderBase):
    def __init__(self, num_heads: int, key_size: int, model_size: int):
        super().__init__(num_heads, key_size, model_size)

    def get_context(  # type: ignore[override]
        self, observation: Observation, embeddings: chex.Array
    ) -> chex.Array:
        return get_context(observation, embeddings)

    def get_transformed_attention_mask(self, attention_mask: chex.Array) -> chex.Array:
        return attention_mask
