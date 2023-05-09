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

from jumanji.environments.routing.connector import Connector, Observation
from jumanji.environments.routing.connector.utils import (
    get_path,
    get_position,
    get_target,
)
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    MultiCategoricalParametricDistribution,
)
from jumanji.training.networks.transformer_block import TransformerBlock


def make_actor_critic_networks_connector(
    connector: Connector,
    transformer_num_blocks: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
    conv_n_channels: int,
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `Connector` environment."""
    num_values = np.asarray(connector.action_spec().num_values)
    parametric_action_distribution = MultiCategoricalParametricDistribution(
        num_values=num_values
    )
    policy_network = make_actor_network_connector(
        num_actions=num_values[0],
        transformer_num_blocks=transformer_num_blocks,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
        conv_n_channels=conv_n_channels,
        env_time_limit=connector.time_limit,
    )
    value_network = make_critic_network_connector(
        num_agents=num_values[0],
        transformer_num_blocks=transformer_num_blocks,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
        conv_n_channels=conv_n_channels,
        env_time_limit=connector.time_limit,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def process_grid(grid: chex.Array, agent_id: chex.Numeric) -> chex.Array:
    """Concatenates two feature maps: the info of the agent and the info about all other agents
    in an indiscernible way (to keep permutation equivariance).
    """
    agent_path = get_path(agent_id)
    agent_target = get_target(agent_id)
    agent_pos = get_position(agent_id)
    agent_grid = jnp.expand_dims(grid, -1)
    agent_mask = (
        (agent_grid == agent_path)
        | (agent_grid == agent_target)
        | (agent_grid == agent_pos)
    )
    # [G, G, 1]
    agent_channel = jnp.where(agent_mask, agent_grid, 0)
    others_channel = jnp.where(agent_mask | (agent_grid == 0), 0, agent_grid)
    # [G, G, 2]
    channels = jnp.concatenate([agent_channel, others_channel], axis=-1)

    return channels.astype(float)


class ConnectorTorso(hk.Module):
    def __init__(
        self,
        transformer_num_blocks: int,
        transformer_num_heads: int,
        transformer_key_size: int,
        transformer_mlp_units: Sequence[int],
        conv_n_channels: int,
        env_time_limit: int,
        num_agents: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.transformer_num_blocks = transformer_num_blocks
        self.transformer_num_heads = transformer_num_heads
        self.transformer_key_size = transformer_key_size
        self.transformer_mlp_units = transformer_mlp_units
        self.model_size = transformer_num_heads * transformer_key_size
        self.num_agents = num_agents
        self.cnn_block = hk.Sequential(
            [
                hk.Conv2D(conv_n_channels, (3, 3), 1, padding="VALID"),
                jax.nn.relu,
                hk.Conv2D(conv_n_channels, (3, 3), 1, padding="VALID"),
                jax.nn.relu,
                hk.Conv2D(conv_n_channels, (3, 3), 1, padding="VALID"),
                jax.nn.relu,
                hk.Conv2D(conv_n_channels, (3, 3), 1, padding="VALID"),
                jax.nn.relu,
                hk.Flatten(),
            ],
            name="cnn_block",
        )
        self.env_time_limit = env_time_limit

    def __call__(self, observation: Observation) -> chex.Array:
        agent_ids = jnp.arange(self.num_agents)
        # (N, B, G, G, 2)
        grid = jax.vmap(process_grid, (None, 0))(observation.grid, agent_ids)
        grid = jnp.transpose(grid, (1, 0, 2, 3, 4))  # (B, N, G, G, 2)

        embeddings = jax.vmap(self.cnn_block)(grid)  # (B, N, H)
        embeddings = self._augment_with_step_count(embeddings, observation.step_count)

        mlp_torso = hk.Sequential(
            [
                hk.nets.MLP((*self.transformer_mlp_units, self.model_size)),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            ],
            name="mlp_torso",
        )
        embeddings = mlp_torso(embeddings)

        mask = self._make_self_attention_mask(observation.action_mask)
        for block_id in range(self.transformer_num_blocks):
            transformer_block = TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_size=self.transformer_key_size,
                mlp_units=self.transformer_mlp_units,
                w_init_scale=2 / self.transformer_num_blocks,
                model_size=self.model_size,
                name=f"self_attention_block_{block_id}",
            )
            embeddings = transformer_block(
                query=embeddings, key=embeddings, value=embeddings, mask=mask
            )
        return embeddings

    def _make_self_attention_mask(self, action_mask: chex.Array) -> chex.Array:
        # Remove agents that can't take any action other than no-op.
        mask = jnp.any(action_mask[..., 1:], axis=-1)
        # Replicate the mask on the query and key dimensions.
        mask = jnp.einsum("...i,...j->...ij", mask, mask)
        # Expand on the head dimension.
        mask = jnp.expand_dims(mask, axis=-3)
        return mask

    def _augment_with_step_count(
        self, embeddings: chex.Array, step_count: chex.Array
    ) -> chex.Array:
        step_count = jnp.asarray(step_count / self.env_time_limit, float)
        num_agents = self.num_agents  # embeddings.shape[-2]
        step_count = jnp.repeat(step_count[:, None], num_agents, axis=-1)[..., None]
        embeddings = jnp.concatenate([embeddings, step_count], axis=-1)
        return embeddings


def make_actor_network_connector(
    num_actions: int,
    transformer_num_blocks: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
    env_time_limit: int,
    conv_n_channels: int,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = ConnectorTorso(
            transformer_num_blocks=transformer_num_blocks,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            conv_n_channels=conv_n_channels,
            env_time_limit=env_time_limit,
            name="policy_torso",
            num_agents=num_actions,
        )
        embeddings = torso(observation)
        logits = hk.nets.MLP((*transformer_mlp_units, num_actions), name="policy_head")(
            embeddings
        )
        logits = jnp.where(observation.action_mask, logits, jnp.finfo(jnp.float32).min)
        return logits

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_critic_network_connector(
    num_agents: int,
    transformer_num_blocks: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
    env_time_limit: int,
    conv_n_channels: int,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = ConnectorTorso(
            transformer_num_blocks=transformer_num_blocks,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            conv_n_channels=conv_n_channels,
            env_time_limit=env_time_limit,
            name="critic_torso",
            num_agents=num_agents,
        )
        embeddings = torso(observation)
        # Sum embeddings over the sequence length (num_agents).
        agent_mask = jnp.any(observation.action_mask[..., 1:], axis=-1)
        embedding = jnp.sum(embeddings, axis=-2, where=agent_mask[..., None])
        value = hk.nets.MLP((*transformer_mlp_units, 1), name="critic_head")(embedding)
        return jnp.squeeze(value, axis=-1)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
