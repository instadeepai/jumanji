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
import jax.numpy as jnp

from jumanji.environments.routing.mmst import MMST, Observation
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    MultiCategoricalParametricDistribution,
)
from jumanji.training.networks.transformer_block import TransformerBlock


def make_actor_critic_networks_mmst(
    mmst: MMST,
    num_transformer_layers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `MMST` environment."""
    num_values = mmst.action_spec().num_values
    parametric_action_distribution = MultiCategoricalParametricDistribution(
        num_values=num_values
    )
    policy_network = make_actor_network_mmst(
        num_transformer_layers=num_transformer_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
        time_limit=mmst.time_limit,
    )
    value_network = make_critic_network_mmst(
        num_transformer_layers=num_transformer_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
        time_limit=mmst.time_limit,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


class MMSTTorso(hk.Module):
    def __init__(
        self,
        num_transformer_layers: int,
        transformer_num_heads: int,
        transformer_key_size: int,
        transformer_mlp_units: Sequence[int],
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_transformer_layers = num_transformer_layers
        self.transformer_num_heads = transformer_num_heads
        self.transformer_key_size = transformer_key_size
        self.transformer_mlp_units = transformer_mlp_units
        self.model_size = transformer_num_heads * transformer_key_size

    def _make_self_attention_mask(self, adj_matrix: chex.Array) -> chex.Array:
        # Expand on the head dimension.
        mask = jnp.expand_dims(adj_matrix, axis=-3)
        return mask

    def embed_nodes(self, nodes: chex.Array) -> chex.Array:
        def get_node_feats(node: chex.Array) -> chex.Array:
            visited = (node != -1) & (node % 2 == 1)
            node_agent = jax.lax.select(node == -1, -1, node // 2)

            return jnp.array([visited, node_agent], dtype=jnp.float32)

        nodes_leaves = jax.vmap(jax.vmap(get_node_feats))(nodes)
        embeddings = hk.Linear(self.model_size, name="node_projection")(nodes_leaves)
        return embeddings

    def embed_agents(self, agents: chex.Array) -> chex.Array:

        embeddings = hk.Linear(self.model_size, name="agent_projection")(agents)
        return embeddings

    def __call__(self, observation: Observation) -> chex.Array:

        batch_size, num_nodes = observation.node_types.shape
        num_agents = observation.positions.shape[1]
        agents_used = jnp.arange(num_agents).reshape(-1, 1)
        agents_used = jnp.repeat(agents_used[jnp.newaxis], batch_size, axis=0)

        agents_nodes = jnp.zeros((batch_size, num_agents, num_nodes), dtype=jnp.float32)
        for i in range(num_agents):
            data = (observation.node_types == i + 1).astype(jnp.float32)
            agents_nodes = agents_nodes.at[:, i].set(data)

        nodes_agents = agents_nodes.transpose((0, 2, 1))

        agents_embeddings = self.embed_agents(
            agents_used.astype(jnp.float32)
        )  # Shape (batch_size, num_agents, 128)

        node_embeddings = self.embed_nodes(
            observation.node_types
        )  # Shape (batch_size, num_nodes, 128)

        mask = self._make_self_attention_mask(
            observation.adj_matrix
        )  # Shape (batch_size, transformer_num_heads, num_nodes, num_nodes)

        # Expand and repeat along the transformer_num_heads axis
        agents_cross_nodes_mask = jnp.expand_dims(
            agents_nodes, axis=1
        )  # Shape (batch_size, 1, num_agents, num_nodes)
        agents_cross_nodes_mask = jnp.repeat(
            agents_cross_nodes_mask, repeats=self.transformer_num_heads, axis=1
        )  # Shape (batch_size, transformer_num_heads, num_agents, num_nodes)

        # Expand and repeat along the transformer_num_heads axis
        nodes_cross_agents_mask = jnp.expand_dims(
            nodes_agents, axis=1
        )  # Shape (batch_size, 1, num_nodes, num_agents)
        nodes_cross_agents_mask = jnp.repeat(
            nodes_cross_agents_mask, repeats=self.transformer_num_heads, axis=1
        )  # Shape (batch_size, transformer_num_heads, num_nodes, num_agents)

        for block_id in range(self.num_transformer_layers):
            # Self-attention on nodes.
            node_embeddings = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=2 / self.num_transformer_layers,
                model_size=self.model_size,
                name=f"self_attention_nodes_block_{block_id}",
            )(node_embeddings, node_embeddings, node_embeddings, mask)

            # Self-attention on agents.
            agents_embeddings = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=2 / self.num_transformer_layers,
                model_size=self.model_size,
                name=f"self_attention_agent_block_{block_id}",
            )(agents_embeddings, agents_embeddings, agents_embeddings)

            # Cross-attention between nodes and agents.
            new_node_embeddings = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=2 / self.num_transformer_layers,
                model_size=self.model_size,
                name=f"cross_attention_node_agent_block_{block_id}",
            )(
                node_embeddings,
                agents_embeddings,
                agents_embeddings,
                nodes_cross_agents_mask,
            )

            # Cross-attention between agents and nodes.
            agents_embeddings = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=2 / self.num_transformer_layers,
                model_size=self.model_size,
                name=f"cross_attention_agent_node_block_{block_id}",
            )(
                agents_embeddings,
                node_embeddings,
                node_embeddings,
                agents_cross_nodes_mask,
            )

            node_embeddings = new_node_embeddings

        batch_indices = jnp.arange(node_embeddings.shape[0])
        current_node_embeddings = jnp.zeros(
            (batch_size, num_agents, self.model_size), dtype=jnp.float32
        )
        for i in range(num_agents):
            data = node_embeddings[batch_indices, observation.positions[:, i]]
            current_node_embeddings = current_node_embeddings.at[:, i].set(data)

        new_embedding = TransformerBlock(
            num_heads=self.transformer_num_heads,
            key_size=self.transformer_key_size,
            mlp_units=self.transformer_mlp_units,
            w_init_scale=2 / self.num_transformer_layers,
            model_size=self.model_size,
            name=f"cross_attention_agent_node_block_{block_id+1}",
        )(agents_embeddings, current_node_embeddings, current_node_embeddings)

        return new_embedding


def make_actor_network_mmst(
    num_transformer_layers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
    time_limit: int,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = MMSTTorso(
            num_transformer_layers=num_transformer_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            name="policy_torso",
        )

        num_agents, num_actions = observation.action_mask.shape[-2:]
        embeddings = torso(observation)  # (B, A, H)
        step_count = observation.step_count[:, None] / time_limit
        step_counts = jnp.tile(
            step_count,
            (
                1,
                num_agents,
            ),
        )[..., None]
        embeddings = jnp.concatenate([embeddings, step_counts], axis=-1)
        logits = hk.nets.MLP((torso.model_size, num_actions), name="policy_head")(
            embeddings
        )  # (B, A, N)
        logits = jnp.where(observation.action_mask, logits, jnp.finfo(jnp.float32).min)
        return logits

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_critic_network_mmst(
    num_transformer_layers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
    time_limit: int,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = MMSTTorso(
            num_transformer_layers=num_transformer_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            name="critic_torso",
        )
        embeddings = torso(observation)
        embedding = jnp.mean(embeddings, axis=-2)
        step_count = jnp.expand_dims(observation.step_count / time_limit, axis=-1)
        embedding = jnp.concatenate([embedding, step_count], axis=-1)
        value = hk.nets.MLP((torso.model_size, 1), name="critic_head")(embedding)
        return jnp.squeeze(value, axis=-1)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
