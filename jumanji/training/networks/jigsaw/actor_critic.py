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

from jumanji.environments.packing.jigsaw import Jigsaw, Observation
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    FactorisedActionSpaceParametricDistribution,
)
from jumanji.training.networks.transformer_block import TransformerBlock


def make_actor_critic_networks_jigsaw(
    jigsaw: Jigsaw,
    num_transformer_layers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `Jigsaw` environment."""
    num_values = np.asarray(jigsaw.action_spec().num_values)
    parametric_action_distribution = FactorisedActionSpaceParametricDistribution(
        action_spec_num_values=num_values
    )
    num_pieces = jigsaw.num_pieces
    policy_network = make_actor_network_jigsaw(
        num_transformer_layers=num_transformer_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
        num_pieces=num_pieces,
    )
    value_network = make_critic_network_jigsaw(
        num_transformer_layers=num_transformer_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
        num_pieces=num_pieces,
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


class JigsawTorso(hk.Module):
    def __init__(
        self,
        num_transformer_layers: int,
        transformer_num_heads: int,
        transformer_key_size: int,
        transformer_mlp_units: Sequence[int],
        num_pieces: int,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.num_transformer_layers = num_transformer_layers
        self.transformer_num_heads = transformer_num_heads
        self.transformer_key_size = transformer_key_size
        self.transformer_mlp_units = transformer_mlp_units
        self.model_size = transformer_num_heads * transformer_key_size
        self.num_pieces = num_pieces

    def __call__(self, observation: Observation) -> Tuple[chex.Array, chex.Array]:
        # Observation.pieces (B, num_pieces, 3, 3)
        # Observation.current_board (B, num_rows, num_cols, 1)

        # Flatten the pieces
        flattened_pieces = jnp.reshape(observation.pieces, (-1, self.num_pieces, 9))
        # Flatten_pieces is of shape (B, num_pieces, 9)

        # Encode the pieces with an MLP
        piece_encoder = hk.nets.MLP(output_sizes=[self.model_size])
        pieces_embedding = jax.vmap(piece_encoder)(flattened_pieces)
        # Pieces_embedding is of shape (B, num_pieces, model_size)

        unet = UNet()
        board_conv_encoding, board_encoding = unet(observation.current_board)
        # board_encoding has shape (B, num_rows-2, num_cols-2)

        # Flatten the board_conv_encoding so it is of shape (B, num_maps, ...)
        board_conv_encoding = jnp.reshape(
            board_conv_encoding,
            (board_conv_encoding.shape[0], board_conv_encoding.shape[1], -1),
        )

        # Cross-attention between pieces_embedding and board_conv_encoding
        for block_id in range(self.num_transformer_layers):
            cross_attention = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                model_size=self.model_size,
                w_init_scale=2 / self.num_transformer_layers,
                name=f"cross_attention_block_{block_id}",
            )
            pieces_embedding = cross_attention(
                query=pieces_embedding,
                key=board_conv_encoding,
                value=board_conv_encoding,
            )

        # Map pieces embedding from (num_pieces, 128) to (num_pieces, num_rotations)
        pieces_head = hk.nets.MLP(output_sizes=[4])
        pieces_embedding = jax.vmap(pieces_head)(pieces_embedding)

        # pieces_embedding has shape (B, num_pieces, num_rotations)
        # board_encoding has shape (B, num_rows-2, num_cols-2) to match
        # the shape of the action mask.
        return pieces_embedding, board_encoding


def make_actor_network_jigsaw(
    num_transformer_layers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
    num_pieces: int,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = JigsawTorso(
            num_transformer_layers=num_transformer_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            num_pieces=num_pieces,
            name="policy_torso",
        )
        pieces_embedding, board_embedding = torso(observation)
        # Outer-product
        outer_product = jnp.einsum(
            "...ij,...kl->...ijkl", pieces_embedding, board_embedding
        )

        logits = jnp.where(
            observation.action_mask, outer_product, jnp.finfo(jnp.float32).min
        )

        logits = logits.reshape(*logits.shape[:-4], -1)
        return logits

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_critic_network_jigsaw(
    num_transformer_layers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
    num_pieces: int,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = JigsawTorso(
            num_transformer_layers=num_transformer_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            num_pieces=num_pieces,
            name="critic_torso",
        )
        pieces_embedding, final_embedding = torso(observation)
        # Flatten and concatenate the pieces embedding and the final embedding
        pieces_embedding_flat = pieces_embedding.reshape(pieces_embedding.shape[0], -1)
        final_embedding_flat = final_embedding.reshape(final_embedding.shape[0], -1)

        # Concatenate along the second dimension (axis=1)
        torso_output = jnp.concatenate(
            [pieces_embedding_flat, final_embedding_flat], axis=-1
        )

        value = hk.nets.MLP((torso.model_size, 1), name="critic_head")(torso_output)
        return jnp.squeeze(value, axis=-1)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
