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
import numpy as np

from jumanji.environments import RobotWarehouse
from jumanji.environments.routing.robot_warehouse.types import Observation
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    MultiCategoricalParametricDistribution,
)
from jumanji.training.networks.transformer_block import TransformerBlock


def make_actor_critic_networks_robot_warehouse(
    robot_warehouse: RobotWarehouse,
    transformer_num_blocks: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `RobotWarehouse` environment."""
    num_values = np.asarray(robot_warehouse.action_spec().num_values)
    parametric_action_distribution = MultiCategoricalParametricDistribution(
        num_values=num_values
    )
    policy_network = make_actor_network(
        time_limit=robot_warehouse.time_limit,
        transformer_num_blocks=transformer_num_blocks,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
    )
    value_network = make_critic_network(
        time_limit=robot_warehouse.time_limit,
        transformer_num_blocks=transformer_num_blocks,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_mlp_units=transformer_mlp_units,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


class RobotWarehouseTorso(hk.Module):
    def __init__(
        self,
        transformer_num_blocks: int,
        transformer_num_heads: int,
        transformer_key_size: int,
        transformer_mlp_units: Sequence[int],
        env_time_limit: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.transformer_num_blocks = transformer_num_blocks
        self.transformer_num_heads = transformer_num_heads
        self.transformer_key_size = transformer_key_size
        self.transformer_mlp_units = transformer_mlp_units
        self.model_size = transformer_num_heads * transformer_key_size
        self.env_time_limit = env_time_limit

    def __call__(self, observation: Observation) -> chex.Array:
        # Shape names:
        # B: batch size
        # N: number of agents
        # O: observation size
        # H: hidden/embedding size
        # (B, N, O)
        _, num_agents, _ = observation.agents_view.shape

        percent_done = observation.step_count / self.env_time_limit
        step = jnp.repeat(percent_done[:, None], num_agents, axis=-1)[..., None]
        agents_view = observation.agents_view

        # join step count and agent view to embed both at the same time
        # (B, N, O + 1)
        obs = jnp.concatenate((agents_view, step), axis=-1)
        # (B, N, O + 1) -> (B, N, H)
        embeddings = hk.Linear(self.model_size)(obs)

        # (B, N, H) -> (B, N, H)
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
                query=embeddings, key=embeddings, value=embeddings
            )
        return embeddings  # (B, N, H)


def make_critic_network(
    time_limit: int,
    transformer_num_blocks: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = RobotWarehouseTorso(
            transformer_num_blocks,
            transformer_num_heads,
            transformer_key_size,
            transformer_mlp_units,
            time_limit,
        )
        embeddings = torso(observation)
        embeddings = jnp.sum(embeddings, axis=-2)

        head = hk.nets.MLP((*transformer_mlp_units, 1), activate_final=False)
        values = head(embeddings)  # (B, 1)
        return jnp.squeeze(values, axis=-1)  # (B,)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_actor_network(
    time_limit: int,
    transformer_num_blocks: int,
    transformer_num_heads: int,
    transformer_key_size: int,
    transformer_mlp_units: Sequence[int],
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        torso = RobotWarehouseTorso(
            transformer_num_blocks,
            transformer_num_heads,
            transformer_key_size,
            transformer_mlp_units,
            time_limit,
        )
        output = torso(observation)

        head = hk.nets.MLP((*transformer_mlp_units, 5), activate_final=False)
        logits = head(output)  # (B, N, 5)
        return jnp.where(observation.action_mask, logits, jnp.finfo(jnp.float32).min)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
