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

from jumanji.environments import Cleaner
from jumanji.environments.routing.cleaner.constants import DIRTY, WALL
from jumanji.environments.routing.cleaner.types import Observation
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    MultiCategoricalParametricDistribution,
)


def make_actor_critic_networks_cleaner(
    cleaner: Cleaner,
    num_conv_channels: Sequence[int],
    policy_layers: Sequence[int],
    value_layers: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `Cleaner` environment."""
    num_values = np.asarray(cleaner.action_spec().num_values)
    parametric_action_distribution = MultiCategoricalParametricDistribution(
        num_values=num_values
    )
    policy_network = make_actor_network(
        num_conv_channels=num_conv_channels,
        mlp_units=policy_layers,
        time_limit=cleaner.time_limit,
    )
    value_network = make_critic_network(
        num_conv_channels=num_conv_channels,
        mlp_units=value_layers,
        time_limit=cleaner.time_limit,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def all_agents_channel(agents_locations: chex.Array, grid: chex.Array) -> chex.Array:
    """Create a channel containing the number of agents per tile.

    Args:
        grid: the maze grid.
        agents_locations: the location of all the agents.

    Returns:
        array: a 2D jax array containing the number of agents on each tile.
    """
    xs, ys = agents_locations[:, 0], agents_locations[:, 1]
    num_agents = agents_locations.shape[0]
    agents_channel = jnp.repeat(jnp.zeros_like(grid)[None, :, :], num_agents, axis=0)
    return jnp.sum(agents_channel.at[jnp.arange(num_agents), xs, ys].set(1), axis=0)


def process_obs_for_critic(observation: Observation) -> chex.Array:
    """Process the `Observation` to be usable by the critic model.

    Args:
        observation: the observation as returned by the environment.

    Returns:
        array: a 3D jax array with 3 channels:
            - Dirty channel: 2D array with 1 for dirty tiles and 0 otherwise.
            - Wall channel: 2D array with 1 for walls and 0 otherwise.
            - Agent channel: 2D array with the number of agents on each tile.
    """
    grid = observation.grid
    dirty_channel = jnp.where(grid == DIRTY, 1, 0)
    wall_channel = jnp.where(grid == WALL, 1, 0)
    agents_channel = all_agents_channel(observation.agents_locations, grid)

    return jnp.stack(
        [dirty_channel, wall_channel, agents_channel], axis=-1, dtype=float
    )


def make_critic_network(
    num_conv_channels: Sequence[int], mlp_units: Sequence[int], time_limit: int
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        # Shapes: B: batch size, N: number of agents, W: grid width, H: grid height
        conv_layers = [
            [
                hk.Conv2D(output_channels, (3, 3)),
                jax.nn.relu,
            ]
            for output_channels in num_conv_channels
        ]
        torso = hk.Sequential(
            [
                *[layer for conv_layer in conv_layers for layer in conv_layer],
                hk.Flatten(),
            ]
        )
        obs = jax.vmap(process_obs_for_critic)(observation)  # (B, H, W, 3)
        embedding = torso(obs)  # (B, W * H)
        normalised_step_count = (
            jnp.expand_dims(observation.step_count, axis=-1) / time_limit
        )  # (B, 1)
        output = jnp.concatenate(
            [embedding, normalised_step_count], axis=-1
        )  # (B, W*H+1)
        values = hk.nets.MLP((*mlp_units, 1), activate_final=False)(output)  # (B, 1)
        return jnp.squeeze(values, axis=-1)  # (B,)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def process_obs_for_actor(observation: Observation) -> chex.Array:
    """Process the `Observation` to be usable by the actor model.

    Args:
        observation: the observation as returned by the environment.

    Returns:
        array: a 4D jax array with 4 channels per agent:
            - Dirty channel: 2D array with 1 for dirty tiles and 0 otherwise.
            - Wall channel: 2D array with 1 for walls and 0 otherwise.
            - Agent channel: 2D array with 1 for the agent position and 0 otherwise.
            - Agents channel: 2D array with the number of agents on each tile.
    """
    grid = observation.grid
    agents_locations = observation.agents_locations

    def create_channels_for_one_agent(agent_location: chex.Array) -> chex.Array:
        dirty_channel = jnp.where(grid == DIRTY, 1, 0)
        wall_channel = jnp.where(grid == WALL, 1, 0)
        agent_channel = (
            jnp.zeros_like(grid).at[agent_location[0], agent_location[1]].set(1)
        )
        agents_channel = all_agents_channel(agents_locations, grid)
        return jnp.stack(
            [dirty_channel, wall_channel, agent_channel, agents_channel],
            axis=-1,
            dtype=float,
        )

    return jax.vmap(create_channels_for_one_agent)(agents_locations)


def make_actor_network(
    num_conv_channels: Sequence[int], mlp_units: Sequence[int], time_limit: int
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        # Shapes: B: batch size, N: number of agents, W: grid width, H: grid height
        conv_layers = [
            [
                hk.Conv2D(output_channels, (3, 3)),
                jax.nn.relu,
            ]
            for output_channels in num_conv_channels
        ]
        torso = hk.Sequential(
            [
                *[layer for conv_layer in conv_layers for layer in conv_layer],
                hk.Flatten(),
            ]
        )
        obs = jax.vmap(process_obs_for_actor)(observation)  # (B, N, H, W, 4)
        embedding = jax.vmap(torso)(obs)  # (B, N, W * H)
        num_agents = obs.shape[1]
        normalised_step_count = jnp.repeat(
            jnp.expand_dims(observation.step_count, axis=(1, 2)) / time_limit,
            num_agents,
            axis=1,
        )  # (B, N, 1)
        output = jnp.concatenate(
            [embedding, normalised_step_count], axis=-1
        )  # (B, N, W*H+1)
        head = hk.nets.MLP((*mlp_units, 4), activate_final=False)
        logits = head(output)  # (B, N, 4)
        return jnp.where(observation.action_mask, logits, jnp.finfo(jnp.float32).min)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
