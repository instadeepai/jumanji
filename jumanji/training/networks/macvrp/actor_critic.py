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

from typing import Sequence, Union

import chex
import haiku as hk
import jax.numpy as jnp
import numpy as np

from jumanji.environments.routing.macvrp import MACVRP, Observation
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    MultiCategoricalParametricDistribution,
)


def make_actor_critic_networks_macvrp(
    macvrp: MACVRP,
    policy_layers: Sequence[int],
    value_layers: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for MACVRP."""
    num_actions = macvrp.action_spec().maximum
    num_vehicles = macvrp.num_vehicles
    parametric_action_distribution = MultiCategoricalParametricDistribution(
        num_values=np.asarray(num_actions).reshape(1)
    )
    policy_network = make_network_macvrp(
        critic=False,
        num_actions=num_actions,
        num_vehicles=num_vehicles,
        mlp_units=policy_layers,
    )
    value_network = make_network_macvrp(
        critic=True,
        num_actions=None,
        num_vehicles=num_vehicles,
        mlp_units=value_layers,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def torso(observation: Observation) -> chex.Array:
    batch_size = observation.nodes.coordinates.shape[0]
    concat_list = [
        observation.nodes.coordinates[:, 0].reshape(batch_size, -1),
        observation.nodes.demands[:, 0].reshape(batch_size, -1),
        observation.windows.start[:, 0].reshape(batch_size, -1),
        observation.windows.end[:, 0].reshape(batch_size, -1),
        observation.coeffs.early[:, 0].reshape(batch_size, -1),
        observation.coeffs.late[:, 0].reshape(batch_size, -1),
        observation.main_vehicles.positions.reshape(batch_size, -1),
        observation.main_vehicles.local_times.reshape(batch_size, -1),
        observation.main_vehicles.capacities.reshape(batch_size, -1),
    ]

    concat_values = jnp.concatenate(concat_list, axis=-1)
    return concat_values


def make_network_macvrp(
    critic: bool,
    num_actions: Union[None, int],
    num_vehicles: int,
    mlp_units: Sequence[int],
) -> FeedForwardNetwork:
    def network_fn(
        observation: Observation,
    ) -> chex.Array:
        # Preprocess the observations using a torso network.
        x = torso(observation)

        if critic:
            head = hk.nets.MLP((*mlp_units, 1), activate_final=False)
            return jnp.squeeze(head(x), axis=-1)
        else:
            assert num_actions is not None

            head = hk.nets.MLP(
                (*mlp_units, num_vehicles * num_actions), activate_final=False
            )
            logits = head(x).reshape(-1, num_vehicles, num_actions)
            logits = jnp.where(
                observation.action_mask, logits, jnp.finfo(jnp.float32).min
            )

            return logits

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
