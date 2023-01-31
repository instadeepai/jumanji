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
import numpy as np

from jumanji.environments.packing.binpack.env import BinPack
from jumanji.environments.packing.binpack.types import Observation
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.binpack.transformer_block import TransformerBlock
from jumanji.training.networks.parametric_distribution import (
    FactorisedActionSpaceParametricDistribution,
)


class BinPackTorso(hk.Module):
    def __init__(
        self,
        transformer_n_blocks: int,
        transformer_mlp_units: Sequence[int],
        transformer_key_size: int,
        transformer_num_heads: int,
    ) -> None:
        super().__init__(name="binpack_torso")
        self.transformer_num_heads = transformer_num_heads
        self.transformer_key_size = transformer_key_size
        item_or_ems_encoding_dim = 1
        self._ems_embedding = hk.nets.MLP(
            [
                transformer_key_size * transformer_num_heads - item_or_ems_encoding_dim,
            ],
            activate_final=False,
        )
        self._item_embedding = hk.nets.MLP(
            [
                transformer_key_size * transformer_num_heads - item_or_ems_encoding_dim,
            ],
            activate_final=False,
        )
        self._ems_or_item_embedding = hk.nets.MLP(
            (item_or_ems_encoding_dim,), activate_final=False
        )

        # Copy w_init from https://theaisummer.com/jax-transformer/.
        self.transformer_blocks = [
            TransformerBlock(
                transformer_mlp_units,
                transformer_key_size,
                transformer_num_heads,
                w_init_scale=2.0 / transformer_n_blocks,
            )
            for _ in range(transformer_n_blocks)
        ]

    def __call__(self, observation: Observation) -> jnp.ndarray:
        # Setup ems embedding
        ems = jax.tree_util.tree_flatten(observation.ems)[0]
        ems = jnp.stack(ems, axis=-1)  # stack the elements of ems into a single vector
        ems_embedding = self._ems_embedding(ems)

        # Setup item embedding
        items = jax.tree_util.tree_flatten(observation.items)[0]
        items = jnp.stack(
            items, axis=-1
        )  # stack the elements of items into a single vector
        item_embedding = self._item_embedding(items)

        # Concatenate & add ems_or_item_embedding
        transformer_input_embedding = jnp.concatenate(
            [ems_embedding, item_embedding], axis=-2
        )
        ems_or_item_embedding = jnp.concatenate(
            [jnp.zeros((*ems.shape[:-1], 1)), jnp.ones((*items.shape[:-1], 1))], axis=-2
        )
        ems_or_item_embedding = self._ems_or_item_embedding(ems_or_item_embedding)
        chex.assert_equal_shape_prefix(
            (transformer_input_embedding, ems_or_item_embedding), -1
        )
        transformer_input_embedding = jnp.concatenate(
            [transformer_input_embedding, ems_or_item_embedding], axis=-1
        )
        # Setup transformer mask
        item_mask = observation.items_mask & ~observation.items_placed
        ems_item_concat_mask = jnp.concatenate(
            [observation.ems_mask, item_mask], axis=-1
        )

        # Now, transformer time!
        out = transformer_input_embedding
        for transformer_block in self.transformer_blocks:
            out = transformer_block(out, ems_item_concat_mask)
        return out


def make_actor_critic_networks_binpack(
    binpack: BinPack,
    policy_layers: Sequence[int],
    value_layers: Sequence[int],
    transformer_n_blocks: int,
    transformer_mlp_units: Sequence[int],
    transformer_key_size: int,
    transformer_num_heads: int,
) -> ActorCriticNetworks:
    """Make actor-critic networks for BinPack."""
    num_ems = binpack.obs_num_ems

    parametric_action_distribution = FactorisedActionSpaceParametricDistribution(
        action_spec_num_values=np.asarray(binpack.action_spec().num_values)
    )
    policy_network = make_binpack_network(
        transformer_n_blocks=transformer_n_blocks,
        transformer_mlp_units=transformer_mlp_units,
        transformer_key_size=transformer_key_size,
        transformer_num_heads=transformer_num_heads,
        head_mlp_units=policy_layers,
        critic=False,
        num_ems=num_ems,
    )
    value_network = make_binpack_network(
        transformer_n_blocks=transformer_n_blocks,
        transformer_mlp_units=transformer_mlp_units,
        transformer_key_size=transformer_key_size,
        transformer_num_heads=transformer_num_heads,
        head_mlp_units=[*value_layers, 1],
        critic=True,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_binpack_network(
    transformer_n_blocks: int,
    transformer_mlp_units: Sequence[int],
    transformer_key_size: int,
    transformer_num_heads: int,
    head_mlp_units: Sequence[int],
    critic: bool,
    num_ems: Optional[int] = None,
) -> FeedForwardNetwork:
    def network_fn(
        observation: Observation,
    ) -> chex.Array:
        binpack_torso = BinPackTorso(
            transformer_n_blocks=transformer_n_blocks,
            transformer_mlp_units=transformer_mlp_units,
            transformer_key_size=transformer_key_size,
            transformer_num_heads=transformer_num_heads,
        )
        x = binpack_torso(observation)

        head_network = hk.nets.MLP(head_mlp_units, activate_final=False)
        x = head_network(x)
        if critic:
            # Value network
            assert head_mlp_units[-1] == 1
            x = jnp.sum(x, axis=-2)
            x = jnp.squeeze(x, axis=-1)
        else:
            # Actor network
            assert head_mlp_units[-1] != 1
            assert num_ems is not None
            ems_enc, item_enc = jnp.split(
                x,
                indices_or_sections=[num_ems],
                axis=-2,
            )
            ems_enc = hk.Linear(head_mlp_units[-1])(ems_enc)
            item_enc = hk.Linear(head_mlp_units[-1])(item_enc)

            logits = jnp.einsum("...ij,...kj->...ik", ems_enc, item_enc)
            # Need to flatten (num_ems, num_items) to (num_ems * num_items,)
            logits = logits.reshape((*logits.shape[:-2], -1))
            logits_mask = observation.action_mask.reshape(
                (*observation.action_mask.shape[:-2], -1)
            )
            x = jnp.where(logits_mask, logits, jnp.finfo(jnp.float32).min)
        return x

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
