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
    num_independent_transformer_layers: int,
    num_joint_transformer_layers: int,
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
        num_independent_transformer_layers=num_independent_transformer_layers,
        num_joint_transformer_layers=num_joint_transformer_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
    )
    value_network = make_critic_network_bin_pack(
        num_independent_transformer_layers=num_independent_transformer_layers,
        num_joint_transformer_layers=num_joint_transformer_layers,
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
        num_independent_transformer_layers: int,
        num_joint_transformer_layers: int,
        transformer_num_heads: int,
        transformer_key_size: int,
        transformer_mlp_units: Sequence[int],
        name: Optional[str] = None,
    ):
        """This module first computes self-attention on the EMSs, in parallel with self-attention
        on the items, then both embeddings are concatenated and the module finishes with
        self-attention on the concatenation of EMSs and items.
        """
        super().__init__(name=name)
        self.num_independent_transformer_layers = num_independent_transformer_layers
        self.num_joint_transformer_layers = num_joint_transformer_layers
        self.transformer_num_heads = transformer_num_heads
        self.transformer_key_size = transformer_key_size
        self.transformer_mlp_units = transformer_mlp_units
        self.model_size = transformer_num_heads * transformer_key_size

    def __call__(self, observation: Observation) -> Tuple[chex.Array, chex.Array]:
        # EMS encoder
        ems_mask = observation.ems_mask
        ems_embeddings = self.self_attention_ems(observation.ems, ems_mask)

        # Item encoder
        items_mask = observation.items_mask & ~observation.items_placed
        items_embeddings = self.self_attention_items(observation.items, items_mask)

        # Decoder
        decoder_mask = jnp.concatenate([ems_mask, items_mask], axis=-1)
        embeddings = jnp.concatenate([ems_embeddings, items_embeddings], axis=-2)
        embeddings = self.self_attention_joint_ems_items(embeddings, decoder_mask)

        return embeddings, decoder_mask

    def self_attention_ems(self, ems: EMS, mask: chex.Array) -> chex.Array:
        mask = self._make_self_attention_mask(mask)

        # Stack the 6 EMS attributes into a single vector [x1, x2, y1, y2, z1, z2].
        ems_leaves = jnp.stack(jax.tree_util.tree_leaves(ems), axis=-1)

        # Projection of the EMSs.
        embeddings = hk.Linear(self.model_size, name="ems_projection")(ems_leaves)
        for block_id in range(self.num_independent_transformer_layers):
            transformer_block = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=1 / self.num_independent_transformer_layers,
                model_size=self.model_size,
                name=f"self_attention_ems_block_{block_id}",
            )
            embeddings = transformer_block(
                query=embeddings, key=embeddings, value=embeddings, mask=mask
            )
        return embeddings

    def self_attention_items(self, items: Item, mask: chex.Array) -> chex.Array:
        mask = self._make_self_attention_mask(mask)

        # Stack the 3 items attributes into a single vector [x_len, y_len, z_len].
        items_leaves = jnp.stack(jax.tree_util.tree_leaves(items), axis=-1)

        # Projection of the EMSs.
        embeddings = hk.Linear(self.model_size, name="item_projection")(items_leaves)
        for block_id in range(self.num_independent_transformer_layers):
            transformer_block = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=1 / self.num_independent_transformer_layers,
                model_size=self.model_size,
                name=f"self_attention_items_block_{block_id}",
            )
            embeddings = transformer_block(
                query=embeddings, key=embeddings, value=embeddings, mask=mask
            )
        return embeddings

    def self_attention_joint_ems_items(
        self, embeddings: chex.Array, mask: chex.Array
    ) -> chex.Array:
        mask = self._make_self_attention_mask(mask)
        for block_id in range(self.num_joint_transformer_layers):
            transformer_block = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=1 / self.num_joint_transformer_layers,
                model_size=self.model_size,
                name=f"self_attention_joint_ems_items_block_{block_id}",
            )
            embeddings = transformer_block(
                query=embeddings, key=embeddings, value=embeddings, mask=mask
            )
        return embeddings

    def _make_self_attention_mask(self, mask: chex.Array) -> chex.Array:
        # Use the same mask for the query and the key.
        mask = jnp.einsum("...i,...j->...ij", mask, mask)
        # Expand on the head dimension.
        mask = jnp.expand_dims(mask, axis=-3)
        return mask


def make_actor_network_bin_pack(
    num_independent_transformer_layers: int,
    num_joint_transformer_layers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = BinPackTorso(
            num_independent_transformer_layers=num_independent_transformer_layers,
            num_joint_transformer_layers=num_joint_transformer_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            name="actor_torso",
        )
        embeddings, _ = torso(observation)
        num_ems = observation.ems_mask.shape[-1]
        ems_embeddings, items_embeddings = jnp.split(embeddings, (num_ems,), axis=-2)
        ems_embeddings = hk.nets.MLP(transformer_mlp_units, name="policy_head_ems")(
            ems_embeddings
        )
        items_embeddings = hk.nets.MLP(transformer_mlp_units, name="policy_head_items")(
            items_embeddings
        )
        joint_embeddings = jnp.einsum(
            "...ek,...ik->...ei", ems_embeddings, items_embeddings
        )
        logits = joint_embeddings / jnp.sqrt(transformer_mlp_units[-1])
        logits = jnp.where(observation.action_mask, logits, jnp.finfo(jnp.float32).min)
        return logits.reshape(*logits.shape[:-2], -1)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_critic_network_bin_pack(
    num_independent_transformer_layers: int,
    num_joint_transformer_layers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = BinPackTorso(
            num_independent_transformer_layers=num_independent_transformer_layers,
            num_joint_transformer_layers=num_joint_transformer_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            name="actor_torso",
        )
        embeddings, decoder_mask = torso(observation)
        # Sum embeddings over the sequence length (EMSs + items).
        embedding = jnp.sum(embeddings, axis=-2, where=decoder_mask[..., None])
        value = hk.nets.MLP((*transformer_mlp_units, 1), name="value_head")(embedding)
        return jnp.squeeze(value, axis=-1)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
