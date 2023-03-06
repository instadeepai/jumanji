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


class TransformerBlock(hk.Module):
    """Adapted from https://theaisummer.com/jax-transformer/"""

    def __init__(
        self,
        mlp_units: Sequence[int],
        key_size: int,
        num_heads: int,
        w_init_scale: float,
    ):
        super(TransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.mlp_units = mlp_units
        self.w_init_scale = w_init_scale

    def __call__(self, x: chex.Array, mask: Optional[chex.Array] = None) -> chex.Array:
        if mask is not None:
            transformer_mask = jnp.einsum("...i,...j->...ij", mask, mask)
            transformer_mask = jnp.expand_dims(
                transformer_mask, axis=-3
            )  # expand on head dim
        else:
            transformer_mask = None
        input_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        attn_out = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            w_init_scale=self.w_init_scale,
        )(input_norm, input_norm, input_norm, transformer_mask)
        h = attn_out + x  # residual connection
        out = hk.nets.MLP((*self.mlp_units, self.num_heads * self.key_size))(h)
        out = out + h  # residual connection
        return out
