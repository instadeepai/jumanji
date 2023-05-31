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

from jumanji.environments.logic.graph_coloring import GraphColoring, Observation
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    CategoricalParametricDistribution,
)
from jumanji.training.networks.transformer_block import TransformerBlock


def make_actor_critic_networks_graph_coloring(
    graph_coloring: GraphColoring,
    num_transformer_layers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `GraphColoring` environment."""
    num_actions = graph_coloring.action_spec().num_values
    parametric_action_distribution = CategoricalParametricDistribution(
        num_actions=num_actions
    )
    policy_network = make_actor_network_graph_coloring(
        num_actions=num_actions,
        num_transformer_layers=num_transformer_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
    )
    value_network = make_critic_network_graph_coloring(
        num_actions=num_actions,
        num_transformer_layers=num_transformer_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


class GraphColoringTorso(hk.Module):
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

    def __call__(self, observation: Observation) -> chex.Array:
        """Transforms the observation using a series of transformations.

        The observation is composed of the following components:
        - observation.action_mask: Represents the colors that are available for the current node.
        - observation.colors: Represents the colors assigned to each node.
            Nodes without an assigned color are marked with -1.
        - observation.current_node_index: Represents the node currently being considered.

        The function first determines which colors are used and which nodes are colored.
        Then it embeds the colors and nodes, and creates a mask for the adjacency matrix.
        It further creates two masks to track the relation between colors and nodes.
        Then the function applies a series of transformer blocks to compute:
            self-attention on nodes and colors and the cross-attention between nodes and colors.
        Finally, it extracts the embedding for the current node and computes a new embedding.

        Args:
            observation: the observation to be transformed.

        Returns:
            new_embedding: the transformed observation.
        """

        batch_size, num_nodes = observation.colors.shape
        colors_used = jnp.isin(observation.colors, jnp.arange(num_nodes))
        color_embeddings = hk.Linear(self.model_size)(
            colors_used[..., None].astype(float)
        )  # Shape (batch_size, num_colors, 128)

        nodes_colored = observation.colors >= 0

        color_embeddings = hk.Linear(self.model_size)(
            colors_used[..., None].astype(float)
        )  # Shape (batch_size, num_colors, 128)

        node_embeddings = hk.Linear(self.model_size)(
            nodes_colored[..., None].astype(float)
        )  # Shape (batch_size, num_colors, 128)

        mask = self._make_self_attention_mask(
            observation.adj_matrix
        )  # Shape (batch_size, 1, num_nodes, num_nodes)

        colors_array = jnp.expand_dims(
            observation.colors, axis=1
        )  # Shape (batch_size, 1, num_nodes)
        color_indices = jnp.arange(observation.colors.shape[1])  # Shape (num_colors,)
        color_indices = color_indices[None, :, None]  # Shape (1, num_colors, 1)

        colors_cross_nodes_mask = (
            colors_array == color_indices
        )  # Shape (batch_size, num_colors, num_nodes)

        # Expand along the transformer_num_heads axis
        colors_cross_nodes_mask = jnp.expand_dims(
            colors_cross_nodes_mask, axis=1
        )  # Shape (batch_size, 1, num_colors, num_nodes)

        colors_array = jnp.expand_dims(
            observation.colors, axis=-1
        )  # Shape (batch_size, num_nodes, 1)
        color_indices = jnp.arange(observation.colors.shape[1])  # Shape (num_colors,)
        color_indices = color_indices[None, None]  # Shape (1, 1, num_colors)

        nodes_cross_colors_mask = (
            colors_array == color_indices
        )  # Shape (batch_size, num_nodes, num_colors)

        # Expand along the transformer_num_heads axis
        nodes_cross_colors_mask = jnp.expand_dims(
            nodes_cross_colors_mask, axis=1
        )  # Shape (batch_size, 1, num_nodes, num_colors)

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

            # Self-attention on colors.
            color_embeddings = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=2 / self.num_transformer_layers,
                model_size=self.model_size,
                name=f"self_attention_colors_block_{block_id}",
            )(color_embeddings, color_embeddings, color_embeddings)

            # Cross-attention between nodes and colors.
            new_node_embeddings = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=2 / self.num_transformer_layers,
                model_size=self.model_size,
                name=f"cross_attention_node_color_block_{block_id}",
            )(
                node_embeddings,
                color_embeddings,
                color_embeddings,
                nodes_cross_colors_mask,
            )

            # Cross-attention between colors and nodes.
            color_embeddings = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=2 / self.num_transformer_layers,
                model_size=self.model_size,
                name=f"cross_attention_color_node_block_{block_id}",
            )(
                color_embeddings,
                node_embeddings,
                node_embeddings,
                colors_cross_nodes_mask,
            )

            node_embeddings = new_node_embeddings

        current_node_embeddings = jnp.take(
            node_embeddings, observation.current_node_index, axis=1
        )
        new_embedding = TransformerBlock(
            num_heads=self.transformer_num_heads,
            key_size=self.transformer_key_size,
            mlp_units=self.transformer_mlp_units,
            w_init_scale=2 / self.num_transformer_layers,
            model_size=self.model_size,
            name=f"cross_attention_color_node_block_{block_id+1}",
        )(color_embeddings, current_node_embeddings, current_node_embeddings)

        return new_embedding


def make_actor_network_graph_coloring(
    num_actions: int,
    num_transformer_layers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = GraphColoringTorso(
            num_transformer_layers=num_transformer_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            name="policy_torso",
        )
        embeddings = torso(observation)  # (B, N, H)
        logits = hk.nets.MLP((torso.model_size, 1), name="policy_head")(
            embeddings
        )  # (B, N, 1)
        logits = jnp.squeeze(logits, axis=-1)  # (B, N)
        logits = jnp.where(observation.action_mask, logits, jnp.finfo(jnp.float32).min)
        return logits

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_critic_network_graph_coloring(
    num_actions: int,
    num_transformer_layers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = GraphColoringTorso(
            num_transformer_layers=num_transformer_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            name="critic_torso",
        )
        embeddings = torso(observation)

        embedding = jnp.mean(embeddings, axis=-2)
        value = hk.nets.MLP((torso.model_size, 1), name="critic_head")(embedding)
        return jnp.squeeze(value, axis=-1)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
