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
    num_rows, num_cols = jigsaw.num_rows, jigsaw.num_cols
    num_pieces = jigsaw.num_pieces
    policy_network = make_actor_network_jigsaw(
        num_transformer_layers=num_transformer_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
        num_pieces=num_pieces,
        num_rows=num_rows,
        num_cols=num_cols,
        board_encoding_dim=10,
    )
    value_network = make_critic_network_jigsaw(
        num_transformer_layers=num_transformer_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
        num_pieces=num_pieces,
        num_rows=num_rows,
        num_cols=num_cols,
        board_encoding_dim=10,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


# Net for down convolution of the board
class SimpleNet(hk.Module):
    def __init__(
        self, num_maps: int, board_encoding_dim: int, name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)
        self.num_maps = num_maps
        self.board_encoding_dim = board_encoding_dim

    def __call__(self, x: chex.Array) -> chex.Array:
        # Assuming x is of shape (batch_size, num_rows, num_cols, 1)
        x = hk.Conv2D(self.num_maps, kernel_shape=3, stride=1, padding="SAME")(x)

        # Flatten
        flat = x.reshape(x.shape[0], -1)

        # Use a linear layer to project to (num_maps, F)
        projection = hk.Linear(self.num_maps * self.board_encoding_dim)(flat)

        # Reshape to desired output shape
        projection = projection.reshape(
            x.shape[0], self.num_maps, self.board_encoding_dim
        )

        return projection


# TODO: Fix this to use convs.
class UpConvNet(hk.Module):
    def __init__(
        self,
        action_mask_num_rows: int,
        action_mask_num_cols: int,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.action_mask_num_rows = action_mask_num_rows
        self.action_mask_num_cols = action_mask_num_cols

    def __call__(self, x: chex.Array) -> chex.Array:
        # Assuming x is of shape (batch_size, num_maps, F)

        # Flatten along last two dimensions
        flat = x.reshape(x.shape[0], -1)

        # Use a linear layer to project to (batch_size, action_mask_num_rows * action_mask_num_cols)
        projection = hk.Linear(self.action_mask_num_rows * self.action_mask_num_cols)(
            flat
        )

        # Reshape to desired output shape
        projection = projection.reshape(
            x.shape[0], self.action_mask_num_rows, self.action_mask_num_cols
        )

        return projection


class JigsawTorso(hk.Module):
    def __init__(
        self,
        num_transformer_layers: int,
        transformer_num_heads: int,
        transformer_key_size: int,
        transformer_mlp_units: Sequence[int],
        num_pieces: int,
        board_encoding_dim: int,
        num_rows: int,
        num_cols: int,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.num_transformer_layers = num_transformer_layers
        self.transformer_num_heads = transformer_num_heads
        self.transformer_key_size = transformer_key_size
        self.transformer_mlp_units = transformer_mlp_units
        self.model_size = transformer_num_heads * transformer_key_size
        self.num_pieces = num_pieces
        self.board_encoding_dim = board_encoding_dim
        self.action_mask_num_rows = num_rows - 3
        self.action_mask_num_cols = num_cols - 3

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

        # Down-convolution on the board, board_encoding_dim must be a multiple of num_maps
        down_conv_net = SimpleNet(
            num_maps=self.board_encoding_dim // 2,
            board_encoding_dim=self.board_encoding_dim,
        )
        board_conv_encoding = down_conv_net(observation.current_board)
        # board_conv_encoding is of shape (B, num_maps, board_encoding_dim)

        # Up convolution on the board
        up_conv_net = UpConvNet(
            action_mask_num_rows=self.action_mask_num_rows,
            action_mask_num_cols=self.action_mask_num_cols,
        )
        board_encoding = up_conv_net(board_conv_encoding)
        # Final_embedding is of shape (B, num_rows, num_cols)

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

        # Map pieces embedding from (num_pieces, 128) to (num_pieces, num_rotations) via mlp
        pieces_head = hk.nets.MLP(output_sizes=[4])
        pieces_embedding = jax.vmap(pieces_head)(pieces_embedding)

        # pieces_embedding has shape (B, num_pieces, num_rotations)
        # board_encoding has shape (B, action_mask_num_rows, action_mask_num_cols)
        return pieces_embedding, board_encoding


def make_actor_network_jigsaw(
    num_transformer_layers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
    num_pieces: int,
    board_encoding_dim: int,
    num_rows: int,
    num_cols: int,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = JigsawTorso(
            num_transformer_layers=num_transformer_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            num_pieces=num_pieces,
            board_encoding_dim=board_encoding_dim,
            num_rows=num_rows,
            num_cols=num_cols,
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
    board_encoding_dim: int,
    num_rows: int,
    num_cols: int,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = JigsawTorso(
            num_transformer_layers=num_transformer_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            num_pieces=num_pieces,
            board_encoding_dim=board_encoding_dim,
            num_rows=num_rows,
            num_cols=num_cols,
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
