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

from jumanji.environments.packing.flat_pack import FlatPack, Observation
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    FactorisedActionSpaceParametricDistribution,
)
from jumanji.training.networks.transformer_block import TransformerBlock


def make_actor_critic_networks_flat_pack(
    flat_pack: FlatPack,
    num_transformer_layers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `FlatPack` environment."""
    num_values = np.asarray(flat_pack.action_spec().num_values)
    parametric_action_distribution = FactorisedActionSpaceParametricDistribution(
        action_spec_num_values=num_values
    )
    num_blocks = flat_pack.num_blocks
    policy_network = make_actor_network_flat_pack(
        num_transformer_layers=num_transformer_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
        num_blocks=num_blocks,
    )
    value_network = make_critic_network_flat_pack(
        num_transformer_layers=num_transformer_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
        num_blocks=num_blocks,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


class UNet(hk.Module):
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name)

    def __call__(self, x: chex.Array) -> chex.Array:
        # Assuming x is of shape (batch_size, num_rows, num_cols)
        # Add a channel dimension
        x = x[..., jnp.newaxis]

        down_1 = hk.Conv2D(2, kernel_shape=2, stride=1, padding="SAME")(x)
        down_2 = hk.Conv2D(4, kernel_shape=2, stride=1, padding="SAME")(down_1)

        # upconv
        up_1 = hk.Conv2DTranspose(2, kernel_shape=2, stride=1, padding="SAME")(down_2)
        up_2 = hk.Conv2DTranspose(1, kernel_shape=2, stride=1, padding="SAME")(up_1)

        # Crop the upconvolved output
        # to be the same size as the action mask.
        up_2 = up_2[:, 1:-1, 1:-1]
        # Remove the channel dimension
        output = jnp.squeeze(up_2, axis=-1)

        # Reshape down_2 to be (B, num_feature_maps, ...)
        down_2 = jnp.transpose(down_2, (0, 3, 1, 2))

        return down_2, output


class FlatPackTorso(hk.Module):
    def __init__(
        self,
        num_transformer_layers: int,
        transformer_num_heads: int,
        transformer_key_size: int,
        transformer_mlp_units: Sequence[int],
        num_blocks: int,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.num_transformer_layers = num_transformer_layers
        self.transformer_num_heads = transformer_num_heads
        self.transformer_key_size = transformer_key_size
        self.transformer_mlp_units = transformer_mlp_units
        self.model_size = transformer_num_heads * transformer_key_size
        self.num_blocks = num_blocks

    def __call__(self, observation: Observation) -> Tuple[chex.Array, chex.Array]:
        # Observation.blocks (B, num_blocks, 3, 3)
        # Observation.current_grid (B, num_rows, num_cols, 1)

        # Flatten the blocks
        flattened_blocks = jnp.reshape(observation.blocks, (-1, self.num_blocks, 9))
        # Flatten_blocks is of shape (B, num_blocks, 9)

        # Encode the blocks with an MLP
        block_encoder = hk.nets.MLP(output_sizes=[self.model_size])
        blocks_embedding = jax.vmap(block_encoder)(flattened_blocks)
        # blocks_embedding is of shape (B, num_blocks, model_size)

        unet = UNet()
        grid_conv_encoding, grid_encoding = unet(observation.current_grid)
        # grid_encoding has shape (B, num_rows-2, num_cols-2)

        # Flatten the grid_conv_encoding so it is of shape (B, num_maps, ...)
        grid_conv_encoding = jnp.reshape(
            grid_conv_encoding,
            (grid_conv_encoding.shape[0], grid_conv_encoding.shape[1], -1),
        )

        # Cross-attention between blocks_embedding and grid_conv_encoding
        for block_id in range(self.num_transformer_layers):
            cross_attention = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                model_size=self.model_size,
                w_init_scale=2 / self.num_transformer_layers,
                name=f"cross_attention_block_{block_id}",
            )
            blocks_embedding = cross_attention(
                query=blocks_embedding,
                key=grid_conv_encoding,
                value=grid_conv_encoding,
            )

        # Map blocks embedding from (num_blocks, 128) to (num_blocks, num_rotations)
        blocks_head = hk.nets.MLP(output_sizes=[4])
        blocks_embedding = jax.vmap(blocks_head)(blocks_embedding)

        # blocks_embedding has shape (B, num_blocks, num_rotations)
        # grid_encoding has shape (B, num_rows-2, num_cols-2) to match
        # the shape of the action mask.
        return blocks_embedding, grid_encoding


def make_actor_network_flat_pack(
    num_transformer_layers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
    num_blocks: int,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = FlatPackTorso(
            num_transformer_layers=num_transformer_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            num_blocks=num_blocks,
            name="policy_torso",
        )
        blocks_embedding, grid_embedding = torso(observation)
        # Outer-product
        outer_product = jnp.einsum(
            "...ij,...kl->...ijkl", blocks_embedding, grid_embedding
        )

        logits = jnp.where(
            observation.action_mask, outer_product, jnp.finfo(jnp.float32).min
        )

        logits = logits.reshape(*logits.shape[:-4], -1)
        return logits

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_critic_network_flat_pack(
    num_transformer_layers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
    num_blocks: int,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = FlatPackTorso(
            num_transformer_layers=num_transformer_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            num_blocks=num_blocks,
            name="critic_torso",
        )
        blocks_embedding, final_embedding = torso(observation)
        # Flatten and concatenate the blocks embedding and the final embedding
        blocks_embedding_flat = blocks_embedding.reshape(blocks_embedding.shape[0], -1)
        final_embedding_flat = final_embedding.reshape(final_embedding.shape[0], -1)

        # Concatenate along the second dimension (axis=1)
        torso_output = jnp.concatenate(
            [blocks_embedding_flat, final_embedding_flat], axis=-1
        )

        value = hk.nets.MLP((torso.model_size, 1), name="critic_head")(torso_output)
        return jnp.squeeze(value, axis=-1)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
