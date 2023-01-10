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

from abc import ABC, abstractmethod
from typing import Union

import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array

from jumanji.environments.packing.knapsack import Observation as KnapsackObservation
from jumanji.environments.routing.cvrp import Observation as CVRPObservation
from jumanji.environments.routing.tsp import Observation as TSPObservation


class EncoderBase(ABC, hk.Module):
    """Transformer-based encoder. Used for TSP, Knapsack and CVRP.
    By default, this is the encoder used by Kool et al. (arXiv:1803.08475) and
    Kwon et al. (arXiv:2010.16011).
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        key_size: int,
        model_size: int,
        expand_factor: int,
        name: str = "encoder",
    ):
        super().__init__(name=name)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.key_size = key_size
        self.model_size = model_size
        self.expand_factor = expand_factor

    def __call__(self, problem: Array) -> Array:
        x = self.get_problem_projection(problem)

        for i in range(self.num_layers):
            mha = hk.MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                model_size=self.model_size,
                w_init_scale=1 / self.num_layers,
                name=f"mha_{i}",
            )
            norm1 = hk.LayerNorm(
                axis=-1,  # should be batch norm according to Kool et al.
                create_scale=True,
                create_offset=True,
                name=f"norm_{i}a",
            )

            x = norm1(x + mha(query=x, key=x, value=x))

            mlp = hk.nets.MLP(
                output_sizes=[self.expand_factor * self.model_size, self.model_size],
                activation=jax.nn.relu,
                activate_final=False,
                name=f"mlp_{i}",
            )
            # should be batch norm according to POMO
            norm2 = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True, name=f"norm_{i}b"
            )
            x = norm2(x + mlp(x))

        return x

    @abstractmethod
    def get_problem_projection(self, problem: Array) -> Array:
        pass


class PolicyDecoderBase(ABC, hk.Module):
    """
    Policy decoder module.
    By default, this is the decoder used by Kool et al. (arXiv:1803.08475) and Kwon et al.
    (arXiv:2010.16011).
    """

    def __init__(
        self,
        num_heads: int,
        key_size: int,
        model_size: int,
        name: str = "policy_decoder",
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.model_size = model_size

    def __call__(
        self,
        observation: Union[TSPObservation, KnapsackObservation, CVRPObservation],
        embeddings: Array,
    ) -> Array:
        context = self.get_context(observation, embeddings)
        mha = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            model_size=self.model_size,
            w_init_scale=1,
            name="mha_dec",
        )

        attention_mask = jnp.expand_dims(observation.action_mask, (-3, -2))
        context = mha(
            query=context,
            key=embeddings,
            value=embeddings,
            mask=self.get_transformed_attention_mask(attention_mask),
        ).squeeze(
            axis=-2
        )  # --> [..., model_size]

        attn_logits = jnp.einsum(
            "...pk,...k->...p", embeddings, context
        )  # --> [..., num_cities/items]
        attn_logits = attn_logits / jnp.sqrt(self.model_size)
        attn_logits = 10 * jnp.tanh(attn_logits)  # clip to [-10,10]
        attn_logits = jnp.where(
            observation.action_mask, attn_logits, jnp.finfo(jnp.float32).min
        )  # Set illegal actions to -inf.

        return attn_logits

    @abstractmethod
    def get_context(
        self,
        observation: Union[TSPObservation, KnapsackObservation, CVRPObservation],
        embeddings: Array,
    ) -> Array:
        pass

    @abstractmethod
    def get_transformed_attention_mask(self, attention_mask: Array) -> Array:
        pass


class CriticDecoderBase(ABC, hk.Module):
    """
    Critic decoder module.
    By default, this is the decoder used by Kool et al. (arXiv:1803.08475) and Kwon et al.
    (arXiv:2010.16011).
    """

    def __init__(
        self,
        num_heads: int,
        key_size: int,
        model_size: int,
        name: str = "critic_decoder",
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.model_size = model_size

    def __call__(
        self,
        observation: Union[TSPObservation, KnapsackObservation, CVRPObservation],
        embeddings: Array,
    ) -> Array:
        context = self.get_context(observation, embeddings)
        mha = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            model_size=self.model_size,
            w_init_scale=1,
            name="mha_dec",
        )

        attention_mask = jnp.expand_dims(observation.action_mask, (-3, -2))
        context = mha(
            query=context,
            key=embeddings,
            value=embeddings,
            mask=self.get_transformed_attention_mask(attention_mask),
        ).squeeze(
            axis=-2
        )  # --> [..., model_size]

        value = hk.Linear(1)(context).squeeze(axis=-1)  # [...,]
        return value

    @abstractmethod
    def get_context(
        self,
        observation: Union[TSPObservation, KnapsackObservation, CVRPObservation],
        embeddings: Array,
    ) -> Array:
        pass

    @abstractmethod
    def get_transformed_attention_mask(self, attention_mask: Array) -> Array:
        pass
