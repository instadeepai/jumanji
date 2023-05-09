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
import math
from typing import Optional, Sequence

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from jumanji.environments.routing.macvrp import MACVRP
from jumanji.environments.routing.macvrp.types import Observation
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    MultiCategoricalParametricDistribution,
)
from jumanji.training.networks.transformer_block import TransformerBlock

class PositionalEncoding(hk.Module):
    def __init__(self, d_model: int, max_len: int = 5000, name: Optional[str] = None):
        super(PositionalEncoding, self).__init__(name=name)
        self.d_model = d_model
        self.max_len = max_len

        # Create matrix of shape (max_len, d_model) representing the positional encoding
        # for an input sequence of length max_len
        pos_enc = jnp.zeros((self.max_len, self.d_model))
        position = jnp.arange(0, self.max_len, dtype=np.float32)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model)
        )
        pos_enc = pos_enc.at[:, 0::2].set(jnp.sin(position * div_term))
        pos_enc = pos_enc.at[:, 1::2].set(jnp.cos(position * div_term))
        pos_enc = pos_enc[None]  # (1, max_len, d_model)
        self.pos_enc = pos_enc

    def __call__(self, embedding: chex.Array) -> chex.Array:
        """Add positional encodings to the embedding for each word in the input sequence.

        Args:
            embedding: input sequence embeddings of shape (B, N, D) where
                B is the batch size, N is input sequence length, and D is
                the embedding dimensionality i.e. d_model.

        Returns:
            Tensor of shape (B, N, D).
        """
        return embedding + self.pos_enc[:, : embedding.shape[1], :]

def make_actor_critic_networks_macvrp(
    macvrp: MACVRP,
    num_vehicles: int,
    num_customers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> ActorCriticNetworks:
    """Create an actor-critic network for the `MACVRP` environment."""
    num_values = np.asarray(macvrp.action_spec().num_values)
    parametric_action_distribution = MultiCategoricalParametricDistribution(
        num_values=num_values
    )
    policy_network = make_actor_network_macvrp(
        num_vehicles=num_vehicles,
        num_customers=num_customers,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
    )
    value_network = make_critic_network_macvrp(
        num_vehicles=num_vehicles,
        num_customers=num_customers,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


class MACVRPTorso(hk.Module):
    def __init__(
        self,
        num_vehicles: int,
        num_customers: int,
        transformer_num_heads: int,
        transformer_key_size: int,
        transformer_mlp_units: Sequence[int],
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_vehicles = num_vehicles
        self.num_customers = num_customers
        self.transformer_num_heads = transformer_num_heads
        self.transformer_key_size = transformer_key_size
        self.transformer_mlp_units = transformer_mlp_units
        self.model_size = transformer_num_heads * transformer_key_size

    def __call__(self, observation: Observation) -> chex.Array:
        # Vehicle encoder
        m_remaining_times = observation.vehicles_remaining_times.astype(float)[
            ..., None
        ]  # (B, M, 1)
        vehicle_embeddings = self.self_attention_vehicles(
            m_remaining_times
        )  # (B, M, D)

        # customer encoder
        o_vehicle_ids = observation.ops_vehicle_ids  # (B, J, O)
        o_durations = observation.ops_durations.astype(float)  # (B, J, O)
        o_mask = observation.ops_mask  # (B, J, O)
        customer_embeddings = jax.vmap(
            self.customer_encoder, in_axes=(-2, -2, -2, None), out_axes=-2
        )(
            o_durations,
            o_vehicle_ids,
            o_mask,
            vehicle_embeddings,
        )  # (B, J, D)
        # Add embedding for no-op
        no_op_emb = hk.Linear(self.model_size)(
            jnp.ones((o_mask.shape[0], 1, 1))
        )  # (B, 1, D)
        customer_embeddings = jnp.concatenate(
            [customer_embeddings, no_op_emb], axis=-2
        )  # (B, J+1, D)

        # Joint (vehicles & customers) self-attention
        embeddings = jnp.concatenate(
            [vehicle_embeddings, customer_embeddings], axis=-2
        )  # (M+J+1, D)
        embeddings = self.self_attention_joint_vehicles_ops(embeddings)
        return embeddings

    def self_attention_vehicles(self, m_remaining_times: chex.Array) -> chex.Array:
        # Projection of vehicles' remaining times
        embeddings = hk.Linear(self.model_size, name="remaining_times_projection")(
            m_remaining_times
        )  # (B, M, D)
        for block_id in range(self.num_vehicles):
            transformer_block = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=1 / self.num_vehicles,
                model_size=self.model_size,
                name=f"self_attention_remaining_times_block_{block_id}",
            )
            embeddings = transformer_block(
                query=embeddings,
                key=embeddings,
                value=embeddings,
            )  # (B, M, D)
        return embeddings

    def customer_encoder(
        self,
        o_durations: chex.Array,
        o_vehicle_ids: chex.Array,
        o_mask: chex.Array,
        m_embedding: chex.Array,
    ) -> chex.Array:
        # Compute mask for self attention between operations
        valid_ops_mask = self._make_self_attention_mask(o_mask)  # (B, 1, O, O)

        # Projection of the operations
        embeddings = hk.Linear(self.model_size, name="durations_projection")(
            o_durations[..., None]
        )  # (B, O, D)

        # Add positional encoding since the operations in each customer must be executed sequentially
        max_num_ops = o_durations.shape[-1]
        pos_encoder = PositionalEncoding(
            d_model=self.model_size, max_len=max_num_ops, name="positional_encoding"
        )
        embeddings = pos_encoder(embeddings)

        # Compute cross attention mask
        num_vehicles = m_embedding.shape[-2]
        cross_attn_mask = self._make_cross_attention_mask(
            o_vehicle_ids, num_vehicles, o_mask
        )  # (B, 1, O, M)

        for block_id in range(self.num_customers):
            # Self attention between the operations in the given customer
            transformer_block = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=1 / self.num_customers,
                model_size=self.model_size,
                name=f"self_attention_ops_durations_block_{block_id}",
            )
            embeddings = transformer_block(
                query=embeddings, key=embeddings, value=embeddings, mask=valid_ops_mask
            )

            # Cross attention between the customer's ops embedding and vehicle embedding
            transformer_block = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=1 / self.num_customers,
                model_size=self.model_size,
                name=f"cross_attention_ops_vehicles_block_{block_id}",
            )
            embeddings = transformer_block(
                query=embeddings,
                key=m_embedding,
                value=m_embedding,
                mask=cross_attn_mask,
            )

        embeddings = jnp.sum(embeddings, axis=-2, where=o_mask[..., None])  # (B, D)

        return embeddings

    def self_attention_joint_vehicles_ops(self, embeddings: chex.Array) -> chex.Array:
        for block_id in range(self.num_customers):
            transformer_block = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=1 / self.num_customers,
                model_size=self.model_size,
                name=f"self_attention_joint_ops_vehicles_block_{block_id}",
            )
            embeddings = transformer_block(
                query=embeddings,
                key=embeddings,
                value=embeddings,
            )
        return embeddings

    def _make_self_attention_mask(self, mask: chex.Array) -> chex.Array:
        # Use the same mask for the query and the key.
        mask = jnp.einsum("...i,...j->...ij", mask, mask)  # (B, O, O)
        # Expand on the head dimension.
        mask = jnp.expand_dims(mask, axis=-3)  # (B, 1, O, O)
        return mask

    def _make_cross_attention_mask(
        self, o_vehicle_ids: chex.Array, num_vehicles: int, o_mask: chex.Array
    ) -> chex.Array:
        # One-hot encode o_vehicle_ids to satisfy permutation equivariance
        o_vehicle_ids = jax.nn.one_hot(o_vehicle_ids, num_vehicles)  # (B, O, M)
        mask = jnp.logical_and(o_vehicle_ids, o_mask[..., None])  # (B, O, M)
        # Expand on the head dimension.
        mask = jnp.expand_dims(mask, axis=-3)  # (B, 1, O, M)
        return mask


def make_actor_network_macvrp(
    num_vehicles: int,
    num_customers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = MACVRPTorso(
            num_vehicles=num_vehicles,
            num_customers=num_customers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            name="actor_torso",
        )
        embeddings = torso(observation)  # (B, M+J+1, D)
        num_vehicles = observation.vehicles_remaining_times.shape[-1]
        vehicle_embeddings, customer_embeddings = jnp.split(
            embeddings, (num_vehicles,), axis=-2
        )
        vehicle_embeddings = hk.Linear(32, name="policy_head_vehicles")(
            vehicle_embeddings
        )
        customer_embeddings = hk.Linear(32, name="policy_head_customers")(customer_embeddings)
        logits = jnp.einsum(
            "...mk,...jk->...mj", vehicle_embeddings, customer_embeddings
        )  # (B, M, J+1)
        logits = jnp.where(observation.action_mask, logits, jnp.finfo(jnp.float32).min)
        return logits

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_critic_network_macvrp(
    num_vehicles: int,
    num_customers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = MACVRPTorso(
            num_vehicles=num_vehicles,
            num_customers=num_customers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            name="critic_torso",
        )
        embeddings = torso(observation)  # (B, M+J+1, D)
        # Sum embeddings over the sequence length (vehicles + customers).
        embedding = jnp.sum(embeddings, axis=-2)
        value = hk.nets.MLP((*transformer_mlp_units, 1), name="value_head")(embedding)
        return jnp.squeeze(value, axis=-1)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)

