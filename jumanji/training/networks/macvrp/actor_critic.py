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
from typing import Optional, Sequence

import chex
import haiku as hk
import jax
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


def make_actor_critic_networks_macvrp(
    macvrp: MACVRP,
    num_vehicles: int,
    num_customers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> ActorCriticNetworks:
    """Create an actor-critic network for the `MACVRP` environment."""

    # Add depot to the number of customers
    num_customers += 1

    num_actions = macvrp.action_spec().maximum
    parametric_action_distribution = MultiCategoricalParametricDistribution(
        num_values=np.asarray(num_actions).reshape(1)
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
        batch_size = observation.nodes.coordinates.shape[0]

        concat_list = [
            observation.main_vehicles.positions.reshape(
                batch_size, self.num_vehicles, -1
            ),
            observation.main_vehicles.local_times.reshape(
                batch_size, self.num_vehicles, -1
            ),
            observation.main_vehicles.capacities.reshape(
                batch_size, self.num_vehicles, -1
            ),
        ]
        o_vehicles = jax.numpy.concatenate(concat_list, axis=-1)

        # Add vehicle ids to be able to break symmetry in
        # the initial observations
        o_vehicle_ids = jax.numpy.identity(self.num_vehicles)

        # Duplicate over the batch dimension
        o_vehicle_ids = jax.numpy.tile(o_vehicle_ids, (batch_size, 1, 1))  # (B, V, V)

        o_vehicles = jax.numpy.concatenate(
            [o_vehicles, o_vehicle_ids], axis=-1
        )  # (B, V, D)

        vehicle_embeddings = self.self_attention_vehicles(o_vehicles)  # (B, V, D)

        # customer encoder
        concat_list = [
            observation.nodes.coordinates[:, 0].reshape(
                batch_size, self.num_customers, -1
            ),
            observation.nodes.demands[:, 0].reshape(batch_size, self.num_customers, -1),
            observation.windows.start[:, 0].reshape(batch_size, self.num_customers, -1),
            observation.windows.end[:, 0].reshape(batch_size, self.num_customers, -1),
            observation.coeffs.early[:, 0].reshape(batch_size, self.num_customers, -1),
            observation.coeffs.late[:, 0].reshape(batch_size, self.num_customers, -1),
        ]

        o_customers = jax.numpy.concatenate(concat_list, axis=-1)

        # (B, C, D)
        customer_embeddings = self.customer_encoder(
            o_customers,
            vehicle_embeddings,
        )

        # Joint (vehicles & customers) self-attention
        embeddings = jax.numpy.concatenate(
            [vehicle_embeddings, customer_embeddings], axis=-2
        )  # (V+C+1, D)

        embeddings = self.self_attention_joint_vehicles_customers(embeddings)
        return embeddings

    def self_attention_vehicles(self, m_remaining_times: chex.Array) -> chex.Array:
        # Projection of vehicles' remaining times
        embeddings = hk.Linear(self.model_size, name="remaining_times_projection")(
            m_remaining_times
        )  # (B, V, D)
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
            )  # (B, V, D)
        return embeddings

    def customer_encoder(
        self,
        o_customers: chex.Array,
        v_embedding: chex.Array,
    ) -> chex.Array:

        # Projection of the operations
        embeddings = hk.Linear(self.model_size, name="o_customer_projections")(
            o_customers
        )  # (B, O, D)

        for block_id in range(self.num_customers):
            # Self attention between the operations in the given customer
            transformer_block = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=1 / self.num_customers,
                model_size=self.model_size,
                name=f"self_attention_customer_block_{block_id}",
            )
            embeddings = transformer_block(
                query=embeddings,
                key=embeddings,
                value=embeddings,
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
                key=v_embedding,
                value=v_embedding,
            )

        return embeddings

    def self_attention_joint_vehicles_customers(
        self, embeddings: chex.Array
    ) -> chex.Array:
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
        embeddings = torso(observation)  # (B, V+C+1, D)

        vehicle_embeddings, customer_embeddings = jax.numpy.split(
            embeddings, (num_vehicles,), axis=-2
        )

        vehicle_embeddings = hk.Linear(32, name="policy_head_vehicles")(
            vehicle_embeddings
        )

        customer_embeddings = hk.Linear(32, name="policy_head_customers")(
            customer_embeddings
        )

        logits = jax.numpy.einsum(
            "...vk,...ck->...vc", vehicle_embeddings, customer_embeddings
        )  # (B, V, C+1)

        logits = jax.numpy.where(
            observation.action_mask, logits, jax.numpy.finfo(jax.numpy.float32).min
        )

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
        embeddings = torso(observation)  # (B, V+C+1, D)
        # Sum embeddings over the sequence length (vehicles + customers).
        embedding = jax.numpy.sum(embeddings, axis=-2)
        value = hk.nets.MLP((*transformer_mlp_units, 1), name="value_head")(embedding)
        return jax.numpy.squeeze(value, axis=-1)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
