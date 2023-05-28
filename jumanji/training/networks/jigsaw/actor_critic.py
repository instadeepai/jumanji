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
        h=10,
        f=5,
    )
    value_network = make_critic_network_jigsaw(
        num_transformer_layers=num_transformer_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
        num_pieces=num_pieces,
        num_rows=num_rows,
        num_cols=num_cols,
        h=10,
        f=10,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


# Net for down convolution of the board
class SimpleNet(hk.Module):
    def __init__(self, num_maps: int, f: int, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.num_maps = num_maps
        self.f = f

    def __call__(self, x: chex.Array) -> chex.Array:
        # Assuming x is of shape (batch_size, num_rows, num_cols, 1)
        x = hk.Conv2D(self.num_maps, kernel_shape=3, stride=1, padding="SAME")(x)

        # Flatten the tensor
        flat = x.reshape(x.shape[0], -1)

        # Use a linear layer to project to (num_maps, F)
        projection = hk.Linear(self.num_maps * self.F)(flat)

        # Reshape to desired output shape
        projection = projection.reshape(x.shape[0], self.num_maps, self.F)

        return projection


# TODO: Fix this to use convs.
class UpConvNet(hk.Module):
    def __init__(
        self, num_rows: int, num_cols: int, name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)
        self.num_rows = num_rows
        self.num_cols = num_cols

    def __call__(self, x: chex.Array) -> chex.Array:
        # Assuming x is of shape (batch_size, num_maps, F)

        # Flatten the tensor along last two dimensions
        flat = x.reshape(x.shape[0], -1)

        # Use a linear layer to project to (batch_size, num_rows * num_cols)
        projection = hk.Linear(self.num_rows * self.num_cols)(flat)

        # Reshape to desired output shape
        projection = projection.reshape(x.shape[0], self.num_rows, self.num_cols)

        return projection


class JigsawTorso(hk.Module):
    def __init__(
        self,
        num_transformer_layers: int,
        transformer_num_heads: int,
        transformer_key_size: int,
        transformer_mlp_units: Sequence[int],
        num_pieces: int,
        h: int,
        f: int,
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
        self.h = h
        self.f = f
        self.num_rows = num_rows - 3
        self.num_cols = num_cols - 3

    def __call__(self, observation: Observation) -> Tuple[chex.Array, chex.Array]:
        # Observation.pieces (B, num_pieces, 3, 3)
        # Observation.current_board (B, num_rows, num_cols, 1)

        # Flatten the pieces
        flattened_pieces = jnp.reshape(observation.pieces, (-1, self.num_pieces, 9))
        # Flatten_pieces is of shape (B, num_pieces, 9)
        jax.debug.print("{x}", x=flattened_pieces.shape)

        # MLP on the pieces
        mlp = hk.nets.MLP(output_sizes=[self.h])
        pieces_embedding = jax.vmap(mlp)(flattened_pieces)
        # Pieces_embedding is of shape (B, num_pieces, H)
        jax.debug.print("{x}", x=pieces_embedding.shape)

        # Down-convolution on the board, F must be a multiple of num_maps
        down_conv_net = SimpleNet(num_maps=self.f // 2, f=self.f)
        middle_embedding = down_conv_net(observation.current_board)
        # Middle_embedding is of shape (B, num_maps, F)
        jax.debug.print("{x}", x=middle_embedding.shape)

        # Up convolution on the board
        up_conv_net = UpConvNet(num_rows=self.num_rows, num_cols=self.num_cols)
        final_embedding = up_conv_net(middle_embedding)
        # Final_embedding is of shape (B, num_rows, num_cols)
        jax.debug.print("{x}", x=final_embedding.shape)

        # Cross-attention between pieces_embedding and middle_embedding
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
                query=pieces_embedding, key=middle_embedding, value=middle_embedding
            )

        # Map pieces embedding from (num_pieces, 128) to (num_pieces, 4) via mlp
        mlp = hk.nets.MLP(output_sizes=[4])
        pieces_embedding = jax.vmap(mlp)(pieces_embedding)
        jax.debug.print("{x}", x=pieces_embedding.shape)

        # Flatten and return
        # return jnp.reshape(outer_product, (-1,))
        return pieces_embedding, final_embedding


def make_actor_network_jigsaw(
    num_transformer_layers: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
    num_pieces: int,
    h: int,
    f: int,
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
            h=h,
            f=f,
            num_rows=num_rows,
            num_cols=num_cols,
            name="policy_torso",
        )
        pieces_embedding, final_embedding = torso(observation)

        # Outer-product
        jax.debug.print("PIECES EMBEDDING {x}", x=pieces_embedding.shape)
        jax.debug.print("FINAL EMBEDDING {x}", x=final_embedding.shape)
        outer_product = jnp.einsum(
            "...j,...kl->...jkl", pieces_embedding, final_embedding
        )
        jax.debug.print("{x}", x=outer_product.shape)

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
    h: int,
    f: int,
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
            h=h,
            f=f,
            num_rows=num_rows,
            num_cols=num_cols,
            name="critic_torso",
        )
        pieces_embedding, final_embedding = torso(observation)
        jax.debug.print("{x}", x=pieces_embedding.shape)
        jax.debug.print("{x}", x=final_embedding.shape)
        # Concatenate the pieces embedding and the final embedding
        torso_output = jnp.concatenate([pieces_embedding, final_embedding], axis=1)

        value = hk.nets.MLP((torso.model_size, 1), name="critic_head")(torso_output)
        return jnp.squeeze(value, axis=-1)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
