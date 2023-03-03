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
    step_limit = rubiks_cube.step_limit
    policy_network = make_network_rubiks_cube(
        cube_embed_dim=cube_embed_dim,
        step_limit=step_limit,
        step_count_embed_dim=step_count_embed_dim,
        dense_layer_dims=dense_layer_dims,
        num_actions=num_actions,
        critic=False,
    )
    value_network = make_network_rubiks_cube(
        cube_embed_dim=cube_embed_dim,
        step_limit=step_limit,
        step_count_embed_dim=step_count_embed_dim,
        dense_layer_dims=dense_layer_dims,
        num_actions=num_actions,
        critic=True,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_network_rubiks_cube(
    cube_embed_dim: int,
    step_limit: int,
    step_count_embed_dim: int,
    dense_layer_dims: Sequence[int],
    num_actions: int,
    critic: bool,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        cube_embedder = hk.Embed(vocab_size=len(Face), embed_dim=cube_embed_dim)
        x = cube_embedder(observation.cube).reshape(observation.cube.shape[0], -1)
        step_count_embedder = hk.Linear(step_count_embed_dim)
        y = step_count_embedder(observation.step_count[:, None] / step_limit)
        dense_layers = hk.nets.MLP(dense_layer_dims)
        output = dense_layers(jnp.concatenate([x, y], axis=-1))
        if critic:
            return jnp.squeeze(hk.Linear(1)(output), axis=-1)
        else:
            return hk.Linear(num_actions)(output)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
