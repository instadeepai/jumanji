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

from jumanji.environments.logic.sliding_tile_puzzle import (
    Observation,
    SlidingTilePuzzle,
)
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    CategoricalParametricDistribution,
)


def make_actor_critic_networks_sliding_tile_puzzle(
    sliding_tile_puzzle: SlidingTilePuzzle,
    policy_layers: Sequence[int],
    value_layers: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `SlidingTilePuzzle` environment."""
    num_actions = sliding_tile_puzzle.action_spec().num_values
    parametric_action_distribution = CategoricalParametricDistribution(
        num_actions=num_actions
    )
    policy_network = make_mlp_network(
        num_outputs=num_actions,
        mlp_units=policy_layers,
    )
    value_network = make_mlp_network(
        num_outputs=1,
        mlp_units=value_layers,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_mlp_network(
    num_outputs: int,
    mlp_units: Sequence[int],
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        # Flatten the puzzle
        flat_puzzle = jnp.reshape(observation.puzzle, (-1,))

        # Pass through MLP
        mlp = hk.nets.MLP((*mlp_units, num_outputs), activate_final=False)
        output = mlp(flat_puzzle)

        if num_outputs == 1:
            return jnp.squeeze(output, axis=-1)
        else:
            return output

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
