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
from math import prod
from typing import Sequence, Tuple, Union

import chex
import haiku as hk
import jax
import jax.numpy as jnp

from jumanji.environments.swarms.search_and_rescue import SearchAndRescue
from jumanji.environments.swarms.search_and_rescue.types import Observation
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    ContinuousActionSpaceNormalTanhDistribution,
)


def make_actor_critic_search_and_rescue(
    search_and_rescue: SearchAndRescue,
    layers: Sequence[int],
) -> ActorCriticNetworks:
    """
    Initialise networks for the search-and-rescue network.

    Note: This network is intended to accept environment observations
        with agent views of shape [n-agents, n-view], but then
        Returns a flattened array of actions for each agent (these
        are reshaped by the wrapped environment).

    Args:
        search_and_rescue: `SearchAndRescue` environment.
        layers: List of hidden layer dimensions.

    Returns:
        Continuous action space MLP action and critic networks.
    """
    n_actions = prod(search_and_rescue.action_spec.shape)
    parametric_action_distribution = ContinuousActionSpaceNormalTanhDistribution(n_actions)
    policy_network = make_actor_network(
        layers=layers, n_agents=search_and_rescue.generator.num_searchers, n_actions=n_actions
    )
    value_network = make_critic_network(layers=layers)

    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def embedding(x: chex.Array) -> chex.Array:
    n_channels = x.shape[-2]
    x = hk.Conv1D(2 * n_channels, 3, data_format="NCW")(x)
    x = jax.nn.relu(x)
    x = hk.MaxPool(2, 2, "SAME", channel_axis=-2)(x)
    return x


def make_critic_network(layers: Sequence[int]) -> FeedForwardNetwork:
    # Shape names:
    # B: batch size
    # N: number of agents
    # C: Observation channels
    # O: observation size

    def network_fn(observation: Observation) -> Union[chex.Array, Tuple[chex.Array, chex.Array]]:
        x = observation.searcher_views  # (B, N, C, O)
        x = hk.vmap(embedding, split_rng=False)(x)
        x = hk.Flatten()(x)
        value = hk.nets.MLP([*layers, 1])(x)  # (B,)
        return jnp.squeeze(value, axis=-1)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_actor_network(layers: Sequence[int], n_agents: int, n_actions: int) -> FeedForwardNetwork:
    # Shape names:
    # B: batch size
    # N: number of agents
    # C: Observation channels
    # O: observation size
    # A: Number of actions

    def log_std_params(x: chex.Array) -> chex.Array:
        return hk.get_parameter("log_stds", shape=x.shape, init=hk.initializers.Constant(0.1))

    def network_fn(observation: Observation) -> Union[chex.Array, Tuple[chex.Array, chex.Array]]:
        x = observation.searcher_views  # (B, N, C, O)
        x = hk.vmap(embedding, split_rng=False)(x)
        x = hk.Flatten()(x)

        means = hk.nets.MLP([*layers, n_agents * n_actions])(x)  # (B, N * A)
        means = hk.Reshape(output_shape=(n_agents, n_actions))(means)  # (B, N, A)

        log_stds = hk.vmap(log_std_params, split_rng=False)(means)  # (B, N, A)

        return means, log_stds

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
