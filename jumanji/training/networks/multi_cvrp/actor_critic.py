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
import jax.numpy as jnp
import numpy as np

from jumanji.environments.routing.multi_cvrp.constants import DEPOT_IDX
from jumanji.environments.routing.multi_cvrp.env import MultiCVRP
from jumanji.environments.routing.multi_cvrp.types import Observation
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    MultiCategoricalParametricDistribution,
)
from jumanji.training.networks.transformer_block import TransformerBlock


def make_actor_critic_networks_multicvrp(
    MultiCVRP: MultiCVRP,  # noqa: N803
    num_vehicles: int,
    num_customers: int,
    num_layers_vehicles: int,
    num_layers_customers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> ActorCriticNetworks:
    """Create an actor-critic network for the `MultiCVRP` environment."""

    # Add depot to the number of customers
    num_customers += 1

    num_actions = MultiCVRP.action_spec().maximum
    parametric_action_distribution = MultiCategoricalParametricDistribution(
        num_values=np.asarray(num_actions).reshape(1)
    )
    policy_network = make_actor_network_multicvrp(
        num_vehicles=num_vehicles,
        num_customers=num_customers,
        num_layers_vehicles=num_layers_vehicles,
        num_layers_customers=num_layers_customers,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
    )
    value_network = make_critic_network_multicvrp(
        num_vehicles=num_vehicles,
        num_customers=num_customers,
        num_layers_vehicles=num_layers_vehicles,
        num_layers_customers=num_layers_customers,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


class MultiCVRPTorso(hk.Module):
    def __init__(
        self,
        num_vehicles: int,
        num_customers: int,
        num_layers_vehicles: int,
        num_layers_customers: int,
        transformer_num_heads: int,
        transformer_key_size: int,
        transformer_mlp_units: Sequence[int],
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_vehicles = num_vehicles
        self.num_customers = num_customers
        self.num_layers_vehicles = num_layers_vehicles
        self.num_layers_customers = num_layers_customers
        self.transformer_num_heads = transformer_num_heads
        self.transformer_key_size = transformer_key_size
        self.transformer_mlp_units = transformer_mlp_units
        self.model_size = transformer_num_heads * transformer_key_size

    def __call__(self, observation: Observation) -> chex.Array:
        # Vehicle encoder
        batch_size = observation.nodes.coordinates.shape[0]

        concat_list = [
            observation.vehicles.coordinates.reshape(batch_size, self.num_vehicles, -1),
            observation.vehicles.local_times.reshape(batch_size, self.num_vehicles, -1),
            observation.vehicles.capacities.reshape(batch_size, self.num_vehicles, -1),
        ]
        o_vehicles = jnp.concatenate(concat_list, axis=-1)

        vehicle_embeddings = self.self_attention_vehicles(o_vehicles)  # (B, V, D)

        # customer encoder
        concat_list = [
            observation.nodes.coordinates.reshape(batch_size, self.num_customers, -1),
            observation.nodes.demands.reshape(batch_size, self.num_customers, -1),
            observation.windows.start.reshape(batch_size, self.num_customers, -1),
            observation.windows.end.reshape(batch_size, self.num_customers, -1),
            observation.coeffs.early.reshape(batch_size, self.num_customers, -1),
            observation.coeffs.late.reshape(batch_size, self.num_customers, -1),
        ]

        o_customers = jnp.concatenate(concat_list, axis=-1)

        # (B, C, D)
        customer_embeddings = self.customer_encoder(
            o_customers,
            vehicle_embeddings,
        )

        vehicle_embeddings = self.vehicle_encoder(  # cross attention
            vehicle_embeddings,
            customer_embeddings,
        )

        return vehicle_embeddings, customer_embeddings

    def self_attention_vehicles(self, m_remaining_times: chex.Array) -> chex.Array:
        # Projection of vehicles' remaining times
        embeddings = hk.Linear(self.model_size, name="remaining_times_projection")(
            m_remaining_times
        )  # (B, V, D)
        for block_id in range(self.num_layers_vehicles):
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

        # Embed the depot differently
        # (B, C, D)
        depot_projection = hk.Linear(self.model_size, name="depot_projection")
        nodes_projection = hk.Linear(self.model_size, name="nodes_projection")
        all_nodes_indices = jnp.arange(o_customers.shape[-2])[None, :, None]
        embeddings = jnp.where(
            all_nodes_indices == DEPOT_IDX,
            depot_projection(o_customers),
            nodes_projection(o_customers),
        )

        for block_id in range(self.num_layers_customers):
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

    def vehicle_encoder(
        self,
        v_embedding: chex.Array,
        c_embedding: chex.Array,
    ) -> chex.Array:

        # Projection of the operations
        embeddings = hk.Linear(self.model_size, name="o_vehicle_projections")(
            v_embedding
        )  # (B, C, D)

        for block_id in range(self.num_layers_vehicles):
            # Self attention between the operations in the given vehicle
            transformer_block = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=1 / self.num_vehicles,
                model_size=self.model_size,
                name=f"self_attention_vehicle_block_{block_id}",
            )
            embeddings = transformer_block(
                query=embeddings,
                key=embeddings,
                value=embeddings,
            )

            # Cross attention between the vehicle's embedding and customer's embedding
            transformer_block = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=1 / self.num_vehicles,
                model_size=self.model_size,
                name=f"cross_attention_ops_vehicles_block_{block_id}",
            )
            embeddings = transformer_block(
                query=embeddings,
                key=c_embedding,
                value=c_embedding,
            )

        return embeddings


def make_actor_network_multicvrp(
    num_vehicles: int,
    num_customers: int,
    num_layers_vehicles: int,
    num_layers_customers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = MultiCVRPTorso(
            num_vehicles=num_vehicles,
            num_customers=num_customers,
            num_layers_vehicles=num_layers_vehicles,
            num_layers_customers=num_layers_customers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            name="actor_torso",
        )
        vehicle_embeddings, customer_embeddings = torso(observation)  # (B, V+C+1, D)

        vehicle_embeddings = hk.Linear(32, name="policy_head_vehicles")(
            vehicle_embeddings
        )

        customer_embeddings = hk.Linear(32, name="policy_head_customers")(
            customer_embeddings
        )

        logits = jnp.einsum(
            "...vk,...ck->...vc", vehicle_embeddings, customer_embeddings
        )  # (B, V, C+1)

        logits = jnp.where(observation.action_mask, logits, jnp.finfo(jnp.float32).min)
        return logits

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_critic_network_multicvrp(
    num_vehicles: int,
    num_customers: int,
    num_layers_vehicles: int,
    num_layers_customers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = MultiCVRPTorso(
            num_vehicles=num_vehicles,
            num_customers=num_customers,
            num_layers_vehicles=num_layers_vehicles,
            num_layers_customers=num_layers_customers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            name="critic_torso",
        )
        vehicle_embeddings, customer_embeddings = torso(observation)  # (B, V+C+1, D)

        # Concatenate the embeddings of the vehicles and customers.
        embeddings = jnp.concatenate(
            [vehicle_embeddings, customer_embeddings], axis=-2
        )  # (B, V+C+1, D)

        # Sum embeddings over the sequence length (vehicles + customers).
        embedding = jnp.mean(embeddings, axis=-2)
        value = hk.nets.MLP((*transformer_mlp_units, 1), name="value_head")(embedding)
        return jnp.squeeze(value, axis=-1)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
