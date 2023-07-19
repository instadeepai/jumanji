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
import jax
import jax.numpy as jnp
import numpy as np

from jumanji.environments.packing.bin_pack import BinPack, Observation
from jumanji.environments.packing.bin_pack.types import EMS, Item
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    FactorisedActionSpaceParametricDistribution,
)
from jumanji.training.networks.transformer_block import TransformerBlock


def make_actor_critic_networks_bin_pack(
    bin_pack: BinPack,
    num_transformer_layers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `BinPack` environment."""
    num_values = np.asarray(bin_pack.action_spec().num_values)
    parametric_action_distribution = FactorisedActionSpaceParametricDistribution(
        action_spec_num_values=num_values
    )
    policy_network = make_actor_network_bin_pack(
        num_transformer_layers=num_transformer_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
    )
    value_network = make_critic_network_bin_pack(
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


class BinPackTorso(hk.Module):
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

    def __call__(self, observation: Observation) -> Tuple[chex.Array, chex.Array]:
        # EMS encoder
        ems_mask = self._make_self_attention_mask(observation.ems_mask)
        ems_embeddings = self.embed_ems(observation.ems)

        # Item encoder
        items_mask = self._make_self_attention_mask(
            observation.items_mask & ~observation.items_placed
        )
        items_embeddings = self.embed_items(observation.items)

        # Decoder
        ems_cross_items_mask = jnp.expand_dims(observation.action_mask, axis=-3)
        items_cross_ems_mask = jnp.expand_dims(
            jnp.moveaxis(observation.action_mask, -1, -2), axis=-3
        )
        for block_id in range(self.num_transformer_layers):
            # Self-attention on EMSs.
            ems_embeddings = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=2 / self.num_transformer_layers,
                model_size=self.model_size,
                name=f"self_attention_ems_block_{block_id}",
            )(ems_embeddings, ems_embeddings, ems_embeddings, ems_mask)

            # Self-attention on items.
            items_embeddings = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=2 / self.num_transformer_layers,
                model_size=self.model_size,
                name=f"self_attention_items_block_{block_id}",
            )(items_embeddings, items_embeddings, items_embeddings, items_mask)

            # Cross-attention EMSs on items.
            new_ems_embeddings = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=2 / self.num_transformer_layers,
                model_size=self.model_size,
                name=f"cross_attention_ems_items_block_{block_id}",
            )(ems_embeddings, items_embeddings, items_embeddings, ems_cross_items_mask)

            # Cross-attention items on EMSs.
            items_embeddings = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=2 / self.num_transformer_layers,
                model_size=self.model_size,
                name=f"cross_attention_items_ems_block_{block_id}",
            )(items_embeddings, ems_embeddings, ems_embeddings, items_cross_ems_mask)
            ems_embeddings = new_ems_embeddings

        return ems_embeddings, items_embeddings

    def embed_ems(self, ems: EMS) -> chex.Array:
        # Stack the 6 EMS attributes into a single vector [x1, x2, y1, y2, z1, z2].
        ems_leaves = jnp.stack(jax.tree_util.tree_leaves(ems), axis=-1)
        # Projection of the EMSs.
        embeddings = hk.Linear(self.model_size, name="ems_projection")(ems_leaves)
        return embeddings

    def embed_items(self, items: Item) -> chex.Array:
        # Stack the 3 items attributes into a single vector [x_len, y_len, z_len].
        items_leaves = jnp.stack(jax.tree_util.tree_leaves(items), axis=-1)
        # Projection of the EMSs.
        embeddings = hk.Linear(self.model_size, name="item_projection")(items_leaves)
        return embeddings

    def _make_self_attention_mask(self, mask: chex.Array) -> chex.Array:
        # Use the same mask for the query and the key.
        mask = jnp.einsum("...i,...j->...ij", mask, mask)
        # Expand on the head dimension.
        mask = jnp.expand_dims(mask, axis=-3)
        return mask


def make_actor_network_bin_pack(
    num_transformer_layers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = BinPackTorso(
            num_transformer_layers=num_transformer_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            name="policy_torso",
        )
        ems_embeddings, items_embeddings = torso(observation)

        # Process EMSs differently from items.
        ems_embeddings = hk.Linear(torso.model_size, name="policy_ems_head")(
            ems_embeddings
        )
        items_embeddings = hk.Linear(torso.model_size, name="policy_items_head")(
            items_embeddings
        )

        # Outer-product between the embeddings to obtain logits.
        logits = jnp.einsum("...ek,...ik->...ei", ems_embeddings, items_embeddings)
        logits = jnp.where(observation.action_mask, logits, jnp.finfo(jnp.float32).min)
        return logits.reshape(*logits.shape[:-2], -1)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_critic_network_bin_pack(
    num_transformer_layers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = BinPackTorso(
            num_transformer_layers=num_transformer_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            name="critic_torso",
        )
        ems_embeddings, items_embeddings = torso(observation)

        # Sum embeddings over the sequence length (EMSs or items).
        ems_mask = observation.ems_mask
        ems_embedding = jnp.sum(ems_embeddings, axis=-2, where=ems_mask[..., None])
        items_mask = observation.items_mask & ~observation.items_placed
        items_embedding = jnp.sum(
            items_embeddings, axis=-2, where=items_mask[..., None]
        )
        joint_embedding = jnp.concatenate([ems_embedding, items_embedding], axis=-1)

        value = hk.nets.MLP((torso.model_size, 1), name="critic_head")(joint_embedding)
        return jnp.squeeze(value, axis=-1)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
