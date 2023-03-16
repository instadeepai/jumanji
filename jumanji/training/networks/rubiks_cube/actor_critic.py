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

from typing import Callable, Sequence

import chex
import haiku as hk
import jax.numpy as jnp
import numpy as np

from jumanji.environments.logic.rubiks_cube.constants import Face
from jumanji.environments.logic.rubiks_cube.env import Observation, RubiksCube
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    FactorisedActionSpaceParametricDistribution,
)


def make_actor_critic_networks_rubiks_cube(
    rubiks_cube: RubiksCube,
    cube_embed_dim: int,
    step_count_embed_dim: int,
    dense_layer_dims: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `RubiksCube` environment."""
    action_spec_num_values = np.asarray(rubiks_cube.action_spec().num_values)
    num_actions = int(np.prod(action_spec_num_values))
    parametric_action_distribution = FactorisedActionSpaceParametricDistribution(
        action_spec_num_values=action_spec_num_values
    )
    time_limit = rubiks_cube.time_limit
    policy_network = make_actor_network(
        cube_embed_dim=cube_embed_dim,
        time_limit=time_limit,
        step_count_embed_dim=step_count_embed_dim,
        dense_layer_dims=dense_layer_dims,
        num_actions=num_actions,
    )
    value_network = make_critic_network(
        cube_embed_dim=cube_embed_dim,
        time_limit=time_limit,
        step_count_embed_dim=step_count_embed_dim,
        dense_layer_dims=dense_layer_dims,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_torso_network_fn(
    cube_embed_dim: int,
    time_limit: int,
    step_count_embed_dim: int,
) -> Callable[[Observation], chex.Array]:
    def torso_network_fn(observation: Observation) -> chex.Array:
        # Cube embedding
        cube_embedder = hk.Embed(vocab_size=len(Face), embed_dim=cube_embed_dim)
        cube_embedding = cube_embedder(observation.cube).reshape(
            *observation.cube.shape[:-3], -1
        )

        # Step count embedding
        step_count_embedder = hk.Linear(step_count_embed_dim)
        step_count_embedding = step_count_embedder(
            observation.step_count[:, None] / time_limit
        )

        embedding = jnp.concatenate([cube_embedding, step_count_embedding], axis=-1)
        return embedding

    return torso_network_fn


def make_actor_network(
    cube_embed_dim: int,
    time_limit: int,
    step_count_embed_dim: int,
    dense_layer_dims: Sequence[int],
    num_actions: int,
) -> FeedForwardNetwork:
    torso_network_fn = make_torso_network_fn(
        cube_embed_dim=cube_embed_dim,
        time_limit=time_limit,
        step_count_embed_dim=step_count_embed_dim,
    )

    def network_fn(observation: Observation) -> chex.Array:
        embedding = torso_network_fn(observation)
        logits = hk.nets.MLP((*dense_layer_dims, num_actions))(embedding)
        return logits

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_critic_network(
    cube_embed_dim: int,
    time_limit: int,
    step_count_embed_dim: int,
    dense_layer_dims: Sequence[int],
) -> FeedForwardNetwork:
    torso_network_fn = make_torso_network_fn(
        cube_embed_dim=cube_embed_dim,
        time_limit=time_limit,
        step_count_embed_dim=step_count_embed_dim,
    )

    def network_fn(observation: Observation) -> chex.Array:
        embedding = torso_network_fn(observation)
        value = hk.nets.MLP((*dense_layer_dims, 1))(embedding)
        return jnp.squeeze(value, axis=-1)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
