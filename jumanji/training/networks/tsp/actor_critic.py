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

from typing import Optional, Sequence, Tuple

import chex
import haiku as hk
import jax.numpy as jnp

from jumanji.environments.routing.tsp import TSP, Observation
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    CategoricalParametricDistribution,
)
from jumanji.training.networks.transformer_block import TransformerBlock


def make_actor_critic_networks_tsp(
    tsp: TSP,
    transformer_num_blocks: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
    mean_cities_in_query: bool,
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `TSP` environment."""
    num_actions = tsp.action_spec().num_values
    parametric_action_distribution = CategoricalParametricDistribution(
        num_actions=num_actions
    )
    policy_network = make_actor_network_tsp(
        transformer_num_blocks=transformer_num_blocks,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
        mean_cities_in_query=mean_cities_in_query,
    )
    value_network = make_critic_network_tsp(
        transformer_num_blocks=transformer_num_blocks,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
        mean_cities_in_query=mean_cities_in_query,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


class TSPTorso(hk.Module):
    def __init__(
        self,
        transformer_num_blocks: int,
        transformer_num_heads: int,
        transformer_key_size: int,
        transformer_mlp_units: Sequence[int],
        name: Optional[str] = None,
    ):
        """Linear embedding of all cities' coordinates followed by `transformer_num_blocks` blocks of self
        attention.
        """
        super().__init__(name=name)
        self.transformer_num_blocks = transformer_num_blocks
        self.transformer_num_heads = transformer_num_heads
        self.transformer_key_size = transformer_key_size
        self.transformer_mlp_units = transformer_mlp_units
        self.model_size = transformer_num_heads * transformer_key_size

    def __call__(self, coordinates: chex.Array, mask: chex.Array) -> chex.Array:
        embeddings = hk.Linear(self.model_size, name="coordinates_projection")(
            coordinates
        )
        for block_id in range(self.transformer_num_blocks):
            transformer_block = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=2 / self.transformer_num_blocks,
                model_size=self.model_size,
                name=f"self_attention_block_{block_id}",
            )
            embeddings = transformer_block(
                query=embeddings, key=embeddings, value=embeddings, mask=mask
            )
        return embeddings


def make_actor_network_tsp(
    transformer_num_blocks: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
    mean_cities_in_query: bool,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = TSPTorso(
            transformer_num_blocks=transformer_num_blocks,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            name="actor_torso",
        )
        self_attention_mask, cross_attention_mask = make_tsp_masks(observation)
        embeddings = torso(observation.coordinates, self_attention_mask)
        query = make_tsp_query(observation, embeddings, mean_cities_in_query)
        cross_attention_block = hk.MultiHeadAttention(
            num_heads=transformer_num_heads,
            key_size=transformer_key_size,
            w_init=hk.initializers.VarianceScaling(1.0),
            name="actor_cross_attention_block",
        )
        cross_attention = cross_attention_block(
            query=query,
            value=embeddings,
            key=embeddings,
            mask=cross_attention_mask,
        ).squeeze(axis=-2)
        logits = jnp.einsum("...Tk,...k->...T", embeddings, cross_attention)
        logits = logits / jnp.sqrt(cross_attention_block.model_size)
        logits = 10 * jnp.tanh(logits)  # clip to [-10,10]
        logits = jnp.where(observation.action_mask, logits, jnp.finfo(jnp.float32).min)
        return logits

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_critic_network_tsp(
    transformer_num_blocks: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
    mean_cities_in_query: bool,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = TSPTorso(
            transformer_num_blocks=transformer_num_blocks,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            name="critic_torso",
        )
        self_attention_mask, cross_attention_mask = make_tsp_masks(observation)
        embeddings = torso(observation.coordinates, self_attention_mask)
        query = make_tsp_query(observation, embeddings, mean_cities_in_query)
        cross_attention_block = hk.MultiHeadAttention(
            num_heads=transformer_num_heads,
            key_size=transformer_key_size,
            w_init=hk.initializers.VarianceScaling(1.0),
            name="critic_cross_attention_block",
        )
        cross_attention = cross_attention_block(
            query=query,
            value=embeddings,
            key=embeddings,
            mask=cross_attention_mask,
        ).squeeze(axis=-2)
        values = jnp.einsum("...Tk,...k->...T", embeddings, cross_attention)
        values = values / jnp.sqrt(cross_attention_block.model_size)
        value = values.sum(axis=-1, where=cross_attention_mask.squeeze(axis=(-2, -3)))
        return value

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_tsp_masks(observation: Observation) -> Tuple[chex.Array, chex.Array]:
    """Return the mask of non-visited cities along with the current and initial cities."""
    # Only consider the non-visited cities.
    mask = observation.action_mask
    # Also include the current city, works even when observation.position == -1.
    mask = mask.at[observation.position].set(True)
    # Also include the initial city, works even when observation.trajectory[..., 0] == -1.
    mask = mask.at[observation.trajectory[..., 0]].set(True)

    # Replicate the mask on the query and key dimensions.
    self_attention_mask = jnp.einsum("...i,...j->...ij", mask, mask)
    # Expand on the head dimension.
    self_attention_mask = jnp.expand_dims(self_attention_mask, axis=-3)

    # Expand on the query dimension.
    cross_attention_mask = jnp.expand_dims(mask, axis=-2)
    # Expand on the head dimension.
    cross_attention_mask = jnp.expand_dims(cross_attention_mask, axis=-3)

    return self_attention_mask, cross_attention_mask


def make_tsp_query(
    observation: Observation, embeddings: chex.Array, mean_cities_in_query: bool
) -> chex.Array:
    """Compute the query as a concatenation between the current city, the initial city and
    optionally the mean of all cities.
    """
    # Current city.
    current_city = jnp.take_along_axis(
        embeddings, observation.position[:, None, None], axis=-2
    ).squeeze(axis=-2)
    # if beginning of the episode, the position is not defined (arbitrarily set to -1).
    current_city = jnp.where(
        observation.position[:, None] == -1,
        jnp.zeros_like(current_city),
        current_city,
    )

    # Initial city where one has to return to at the end.
    initial_city = jnp.take_along_axis(
        embeddings, observation.trajectory[:, 0, None, None], axis=-2
    ).squeeze(axis=-2)
    # if beginning of the episode, the position is not defined (arbitrarily set to -1).
    initial_city = jnp.where(
        observation.trajectory[:, 0, None] == -1,
        jnp.zeros_like(initial_city),
        initial_city,
    )

    if mean_cities_in_query:
        mean_cities = jnp.mean(embeddings, axis=-2)
        query = jnp.concatenate([current_city, initial_city, mean_cities], axis=-1)
    else:
        query = jnp.concatenate([current_city, initial_city], axis=-1)
    return jnp.expand_dims(query, axis=-2)
