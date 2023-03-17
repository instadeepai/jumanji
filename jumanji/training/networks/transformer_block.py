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


class TransformerBlock(hk.Module):
    """Transformer block with post layer norm, implementing Attention Is All You Need
    [Vaswani et al., 2016].
    """

    def __init__(
        self,
        num_heads: int,
        key_size: int,
        mlp_units: Sequence[int],
        w_init_scale: float,
        model_size: Optional[int] = None,
        name: Optional[str] = None,
    ):
        """Initialises the transformer block module.

        Args:
            num_heads: number of independent attention heads (H).
            key_size: the size of keys (K) and queries (Q) used in the attention mechanism.
            mlp_units: sequence of MLP layers in the feedforward networks following self-attention.
            w_init_scale: scale to `VarianceScaling` weight initializer.
            model_size: optional size of the output embedding (D'). If None, defaults
                to the key size multiplied by the number of heads (K * H).
            name: optional name for this module.
        """
        super(TransformerBlock, self).__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.mlp_units = mlp_units
        self.w_init = hk.initializers.VarianceScaling(w_init_scale)
        self.model_size = model_size or key_size * num_heads

    def __call__(
        self,
        query: chex.Array,
        key: chex.Array,
        value: chex.Array,
        mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        """Computes in this order:
            - (optionally masked) MHA with queries, keys & values
            - skip connection
            - layer norm
            - MLP
            - skip connection
            - layer norm

        This module broadcasts over zero or more 'batch-like' leading dimensions.

        Args:
            query: embeddings sequence used to compute queries; shape [..., T', D_q].
            key: embeddings sequence used to compute keys; shape [..., T, D_k].
            value: embeddings sequence used to compute values; shape [..., T, D_v].
            mask: optional mask applied to attention weights; shape [..., H=1, T', T].

        Returns:
            A new sequence of embeddings, consisting of a projection of the
                attention-weighted value projections; shape [..., T', D'].
        """

        # Multi-head attention and residual connection
        mha = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            w_init=self.w_init,
            model_size=self.model_size,
        )
        h = mha(query, key, value, mask) + query
        h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)

        # MLP and residual connection
        mlp = hk.nets.MLP((*self.mlp_units, self.model_size), activate_final=True)
        out = mlp(h) + h
        out = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(out)

        return out
