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
    hidden_size: int,
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
        hidden_size=hidden_size,
    )
    value_network = make_critic_network_flat_pack(
        num_transformer_layers=num_transformer_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
        num_blocks=num_blocks,
        hidden_size=hidden_size,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


class UNet(hk.Module):
    """A simple module based on the UNet architecture.
    Please note that all shapes assume an 11x11 grid observation to match the
    default grid size of the FlatPack environment.
    """

    def __init__(
        self,
        model_size: int,
        name: Optional[str] = None,
        hidden_size: int = 8,
    ) -> None:
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.model_size = model_size

    def __call__(self, grid_observation: chex.Array) -> chex.Array:
        # Grid observation is of shape (B, num_rows, num_cols)

        # Add a channel dimension
        grid_observation = grid_observation[..., jnp.newaxis]

        # Down colvolve with strided convolutions
        down_1 = hk.Conv2D(32, kernel_shape=3, stride=2, padding="SAME")(
            grid_observation
        )
        down_1 = jax.nn.relu(down_1)  # (B, 6, 6, 32)
        down_2 = hk.Conv2D(32, kernel_shape=3, stride=2, padding="SAME")(down_1)
        down_2 = jax.nn.relu(down_2)  # (B, 3, 3, 32)

        # Up convolve
        up_1 = hk.Conv2DTranspose(32, kernel_shape=3, stride=2, padding="SAME")(down_2)
        up_1 = jax.nn.relu(up_1)  # (B, 6, 6, 32)
        up_1 = jnp.concatenate([up_1, down_1], axis=-1)
        up_2 = hk.Conv2DTranspose(32, kernel_shape=3, stride=2, padding="SAME")(up_1)
        up_2 = jax.nn.relu(up_2)  # (B, 12, 12, 32)
        up_2 = up_2[:, :-1, :-1]
        up_2 = jnp.concatenate(
            [up_2, grid_observation], axis=-1
        )  # (B, num_rows, num_cols, 33)

        output = hk.Conv2D(self.hidden_size, kernel_shape=1, stride=1, padding="SAME")(
            up_2
        )

        # Crop the upconvolved output to be the same size as the action mask.
        output = output[:, 1:-1, 1:-1]  # (B, num_rows-2, num_cols-2, hidden_size)

        # Flatten down_2 to be (B, ...)
        grid_conv_encoding = jnp.reshape(
            down_2,
            (down_2.shape[0], -1),
        )

        # Linear mapping to transformer model size.
        grid_conv_encoding = hk.Linear(self.model_size)(
            grid_conv_encoding
        )  # (B, model_size)

        return grid_conv_encoding, output


class FlatPackTorso(hk.Module):
    def __init__(
        self,
        num_transformer_layers: int,
        transformer_num_heads: int,
        transformer_key_size: int,
        transformer_mlp_units: Sequence[int],
        num_blocks: int,
        hidden_size: int,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.num_transformer_layers = num_transformer_layers
        self.transformer_num_heads = transformer_num_heads
        self.transformer_key_size = transformer_key_size
        self.transformer_mlp_units = transformer_mlp_units
        self.model_size = transformer_num_heads * transformer_key_size
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size

    def __call__(self, observation: Observation) -> Tuple[chex.Array, chex.Array]:
        # observation.blocks (B, num_blocks, 3, 3)
        # observation.grid (B, num_rows, num_cols)

        # Flatten the blocks
        flattened_blocks = jnp.reshape(
            observation.blocks, (-1, self.num_blocks, 9)
        )  # (B, num_blocks, 9)

        # Encode the blocks with an MLP
        block_encoder = hk.nets.MLP(output_sizes=[self.model_size])
        blocks_embedding = jax.vmap(block_encoder)(
            flattened_blocks
        )  # (B, num_blocks, model_size)

        unet = UNet(hidden_size=self.hidden_size, model_size=self.model_size)
        grid_conv_encoding, grid_encoding = unet(
            observation.grid
        )  # (B, model_size), (B, num_rows-2, num_cols-2, hidden_size)

        for block_id in range(self.num_transformer_layers):

            (
                self_attention_mask,  # (B, 1, num_blocks, num_blocks)
                cross_attention_mask,  # (B, 1, num_blocks, 1)
            ) = make_flatpack_masks(observation)

            self_attention = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                model_size=self.model_size,
                w_init_scale=2 / self.num_transformer_layers,
                name=f"self_attention_block_{block_id}",
            )
            blocks_embedding = self_attention(
                query=blocks_embedding,
                key=blocks_embedding,
                value=blocks_embedding,
                mask=self_attention_mask,
            )

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
                mask=cross_attention_mask,
            )

        # Map blocks embedding from (num_blocks, 128) to (num_blocks, num_rotations, hidden_size)
        blocks_head = hk.nets.MLP(output_sizes=[4 * self.hidden_size])
        blocks_embedding = jax.vmap(blocks_head)(blocks_embedding)
        blocks_embedding = jnp.reshape(
            blocks_embedding, (-1, self.num_blocks, 4, self.hidden_size)
        )

        return blocks_embedding, grid_encoding


def make_actor_network_flat_pack(
    num_transformer_layers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
    num_blocks: int,
    hidden_size: int,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = FlatPackTorso(
            num_transformer_layers=num_transformer_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            num_blocks=num_blocks,
            hidden_size=hidden_size,
            name="policy_torso",
        )
        blocks_embedding, grid_embedding = torso(observation)
        outer_product = jnp.einsum(
            "...ijh,...klh->...ijkl", blocks_embedding, grid_embedding
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
    hidden_size: int,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = FlatPackTorso(
            num_transformer_layers=num_transformer_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            num_blocks=num_blocks,
            hidden_size=hidden_size,
            name="critic_torso",
        )

        (
            blocks_embedding,  # (B, num_blocks, 4, hidden_size)
            grid_embedding,  # (B, num_rows-2, num_cols-2, hidden_size)
        ) = torso(observation)

        # Flatten the blocks embedding
        blocks_embedding = jnp.reshape(
            blocks_embedding,
            (*blocks_embedding.shape[0:2], -1),
        )

        # Sum over blocks for permutation invariance
        blocks_embedding = jnp.sum(blocks_embedding, axis=1)  # (B, 4*hidden_size)

        # Flatten grid embedding while keeping batch dimension
        grid_embedding = jnp.reshape(  # (B, hidden_size * num_rows-2 * num_cols-2)
            grid_embedding,
            (grid_embedding.shape[0], -1),
        )

        grid_embedding = hk.Linear(blocks_embedding.shape[-1])(grid_embedding)
        grid_embedding = jax.nn.relu(grid_embedding)

        # Concatenate along the second dimension
        torso_output = jnp.concatenate([blocks_embedding, grid_embedding], axis=-1)

        value = hk.Linear(1)(torso_output)

        return jnp.squeeze(value, axis=-1)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_flatpack_masks(observation: Observation) -> Tuple[chex.Array, chex.Array]:
    """Return:
    - self_attention_mask: mask of non-placed blocks.
    - cross_attention_mask: action mask, i.e. blocks that can be placed.
    """

    mask = jnp.any(observation.action_mask, axis=(2, 3, 4))

    # Replicate the mask on the query and key dimensions.
    self_attention_mask = jnp.einsum("...i,...j->...ij", mask, mask)
    # Expand on the head dimension.
    self_attention_mask = jnp.expand_dims(self_attention_mask, axis=-3)

    # Expand on the query dimension.
    cross_attention_mask = jnp.expand_dims(mask, axis=-2)
    # Expand on the head dimension.
    cross_attention_mask = jnp.expand_dims(cross_attention_mask, axis=-1)

    return self_attention_mask, cross_attention_mask
