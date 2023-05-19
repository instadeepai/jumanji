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

from typing import Sequence

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from jumanji.environments import Rware
from jumanji.environments.routing.rware.types import Observation
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    MultiCategoricalParametricDistribution,
)


def make_actor_critic_networks_rware(
    rware: Rware,
    agents_view_embed_dim: int,
    step_count_embed_dim: int,
    policy_layers: Sequence[int],
    value_layers: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `Rware` environment."""
    num_values = np.asarray(rware.action_spec().num_values)
    parametric_action_distribution = MultiCategoricalParametricDistribution(
        num_values=num_values
    )
    policy_network = make_actor_network(
        agents_view_embed_dim=agents_view_embed_dim,
        mlp_units=policy_layers,
        time_limit=rware.time_limit,
        step_count_embed_dim=step_count_embed_dim,
    )
    value_network = make_critic_network(
        agents_view_embed_dim=agents_view_embed_dim,
        mlp_units=value_layers,
        time_limit=rware.time_limit,
        step_count_embed_dim=step_count_embed_dim,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_critic_network(
    agents_view_embed_dim: int,
    mlp_units: Sequence[int],
    time_limit: int,
    step_count_embed_dim: int,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        # Shapes: B: batch size, N: number of agents, P: agents embed dim, S: step embed dim
        # agents_view embedding
        batch_size = observation.agents_view.shape[0]
        agents_view_embedder = hk.Linear(agents_view_embed_dim)
        agents_view_embedding = agents_view_embedder(
            observation.agents_view.reshape(batch_size, -1).astype(jnp.float32)
        )  # (B, P)

        # step count embedding
        step_count_embedder = hk.Linear(step_count_embed_dim)
        step_count_embedding = step_count_embedder(
            observation.step_count[:, None] / time_limit
        )  # (B, S)
        output = jnp.concatenate(
            [agents_view_embedding, step_count_embedding], axis=-1
        )  # (B, P+S)
        values = hk.nets.MLP((*mlp_units, 1), activate_final=False)(output)  # (B, 1)
        return jnp.squeeze(values, axis=-1)  # (B,)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_actor_network(
    agents_view_embed_dim: int,
    mlp_units: Sequence[int],
    time_limit: int,
    step_count_embed_dim: int,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        # Shapes: B: batch size, N: number of agents, P: agents embed dim, S: step embed dim
        # agents_view embedding
        per_agent_view_embedder = hk.Linear(agents_view_embed_dim)
        per_agent_view_embedding = jax.vmap(per_agent_view_embedder)(
            observation.agents_view.astype(jnp.float32)
        )  # (B, N, P)
        num_agents = per_agent_view_embedding.shape[1]
        normalised_step_count = jnp.repeat(
            jnp.expand_dims(observation.step_count, axis=(1, 2)) / time_limit,
            num_agents,
            axis=1,
        )  # (B, N, 1)
        # step count embedding
        step_count_embedder = hk.Linear(step_count_embed_dim)
        step_count_embedding = jax.vmap(step_count_embedder)(
            normalised_step_count
        )  # (B, N, S)
        output = jnp.concatenate(
            [per_agent_view_embedding, step_count_embedding], axis=-1
        )  # (B, N, P+S)
        head = hk.nets.MLP((*mlp_units, 5), activate_final=False)
        logits = head(output)  # (B, N, 5)
        return jnp.where(observation.action_mask, logits, jnp.finfo(jnp.float32).min)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
