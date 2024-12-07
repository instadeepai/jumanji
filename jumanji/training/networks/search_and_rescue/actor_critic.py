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
    policy_network = make_actor_network(layers=layers, n_actions=n_actions)
    value_network = make_critic_network(layers=layers)

    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_critic_network(layers: Sequence[int]) -> FeedForwardNetwork:
    # Shape names:
    # B: batch size
    # N: number of agents
    # O: observation size

    def network_fn(observation: Observation) -> Union[chex.Array, Tuple[chex.Array, chex.Array]]:
        views = observation.searcher_views  # (B, N, O)
        batch_size = views.shape[0]
        views = views.reshape(batch_size, -1)  # (B, N * O)
        value = hk.nets.MLP([*layers, 1])(views)  # (B,)
        return jnp.squeeze(value, axis=-1)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_actor_network(layers: Sequence[int], n_actions: int) -> FeedForwardNetwork:
    # Shape names:
    # B: batch size
    # N: number of agents
    # O: observation size
    # A: Number of actions

    def network_fn(observation: Observation) -> Union[chex.Array, Tuple[chex.Array, chex.Array]]:
        views = observation.searcher_views  # (B, N, O)
        batch_size = views.shape[0]
        n_agents = views.shape[1]
        views = views.reshape((batch_size, -1))  # (B, N * 0)
        means = hk.nets.MLP([*layers, n_agents * n_actions])(views)  # (B, N * A)
        means = means.reshape(batch_size, n_agents, n_actions)  # (B, N, A)

        log_stds = hk.get_parameter(
            "log_stds", shape=(n_agents * n_actions,), init=hk.initializers.Constant(0.1)
        )  # (N * A,)
        log_stds = jnp.broadcast_to(log_stds, (batch_size, n_agents * n_actions))  # (B, N * A)
        log_stds = log_stds.reshape(batch_size, n_agents, n_actions)  # (B, N, A)

        return means, log_stds

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
