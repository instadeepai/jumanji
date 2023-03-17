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

from jumanji.environments.routing.maze import Maze, Observation
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    CategoricalParametricDistribution,
)


def make_actor_critic_networks_maze(
    maze: Maze,
    num_channels: Sequence[int],
    policy_layers: Sequence[int],
    value_layers: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `Maze` environment."""
    num_actions = np.asarray(maze.action_spec().num_values)
    parametric_action_distribution = CategoricalParametricDistribution(
        num_actions=num_actions
    )
    policy_network = make_network_maze(
        maze=maze,
        critic=False,
        conv_n_channels=num_channels,
        mlp_units=policy_layers,
        num_actions=num_actions,
    )
    value_network = make_network_maze(
        maze=maze,
        critic=True,
        conv_n_channels=num_channels,
        mlp_units=value_layers,
        num_actions=num_actions,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def process_observation(observation: Observation) -> chex.Array:
    """Add the agent and the target to the walls array."""
    agent = 2
    target = 3
    obs = observation.walls.astype(float)
    obs = obs.at[tuple(observation.agent_position)].set(agent)
    obs = obs.at[tuple(observation.target_position)].set(target)
    return jnp.expand_dims(obs, axis=-1)  # Adding a channels axis.


def make_network_maze(
    maze: Maze,
    critic: bool,
    conv_n_channels: Sequence[int],
    mlp_units: Sequence[int],
    num_actions: int,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        conv_layers = [
            [
                hk.Conv2D(output_channels, (3, 3)),
                jax.nn.relu,
            ]
            for output_channels in conv_n_channels
        ]
        torso = hk.Sequential(
            [
                *[layer for conv_layer in conv_layers for layer in conv_layer],
                hk.Flatten(),
            ]
        )

        obs = jax.vmap(process_observation)(observation)  # (B, G, G, 1)
        embedding = torso(obs)  # (B, H)
        normalised_step_count = (
            jnp.expand_dims(observation.step_count, axis=-1) / maze.time_limit
        )  # (B, 1)
        output = jnp.concatenate(
            [embedding, normalised_step_count], axis=-1
        )  # (B, H+1)

        if critic:
            head = hk.nets.MLP((*mlp_units, 1), activate_final=False)
            return jnp.squeeze(head(output), axis=-1)
        else:
            head = hk.nets.MLP((*mlp_units, num_actions), activate_final=False)
            logits = head(output)
            return jnp.where(
                observation.action_mask, logits, jnp.finfo(jnp.float32).min
            )

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
