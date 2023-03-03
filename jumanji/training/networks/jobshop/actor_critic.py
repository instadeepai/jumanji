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

from jumanji.environments.packing.jobshop import JobShop
from jumanji.environments.packing.jobshop.types import Observation
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    MultiCategoricalParametricDistribution,
)


def make_actor_critic_networks_jobshop(
    jobshop: JobShop,
    policy_layers: Sequence[int],
    value_layers: Sequence[int],
    operations_layers: Sequence[int],
    machines_layers: Sequence[int],
) -> ActorCriticNetworks:
    """Create an actor-critic network for the `JobShop` environment.

    Args:
        jobshop: the instance of the `JobShop` environment.
        policy_layers: sequence of no. of layers in the MLP in the head of the
            critic network.
        value_layers: sequence of no. of layers in the MLP in the head of the
            actor network.
        operations_layers: number of nodes in the MLP that learns the embedding
            for the operations.
        machines_layers: number of nodes in the MLP that learns the embedding
            for the machines.

    Returns:
        An actor-critic network.
    """
    num_values = np.asarray(jobshop.action_spec().num_values)
    parametric_action_distribution = MultiCategoricalParametricDistribution(
        num_values=num_values
    )
    policy_network = make_actor_network_jobshop(
        mlp_units=policy_layers,
        operations_layers=operations_layers,
        machines_layers=machines_layers,
    )
    value_network = make_critic_network_jobshop(
        mlp_units=value_layers,
        operations_layers=operations_layers,
        machines_layers=machines_layers,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_actor_network_jobshop(
    mlp_units: Sequence[int],
    operations_layers: Sequence[int],
    machines_layers: Sequence[int],
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        network_torso = NetworkTorso(operations_layers, machines_layers)
        x = network_torso(observation)

        num_jobs = observation.ops_machine_ids.shape[1]
        num_machines = observation.machines_job_ids.shape[1]
        batch_size = observation.machines_job_ids.shape[0]
        head = hk.nets.MLP(
            (*mlp_units, num_machines * (num_jobs + 1)), activate_final=False
        )
        logits = head(x).reshape(batch_size, num_machines, num_jobs + 1)
        logits = jnp.where(observation.action_mask, logits, jnp.finfo(jnp.float32).min)
        return logits

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


def make_critic_network_jobshop(
    mlp_units: Sequence[int],
    operations_layers: Sequence[int],
    machines_layers: Sequence[int],
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        network_torso = NetworkTorso(operations_layers, machines_layers)
        x = network_torso(observation)

        head = hk.nets.MLP((*mlp_units, 1), activate_final=False)
        return jnp.squeeze(head(x), axis=-1)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)


class NetworkTorso(hk.Module):
    """Common network torso that is used by both the actor network an the critic
    network. This torso consists of two components:
        - Learning an embedding for the operations.
        - Learning an embedding for the status of the machines.
    """

    def __init__(
        self,
        operations_layers: Sequence[int],
        machines_layers: Sequence[int],
    ):
        super(NetworkTorso, self).__init__()
        self.operations_layers = operations_layers
        self.machines_layers = machines_layers

    def __call__(self, observation: Observation) -> jnp.array:
        """Call the network torso module. The following attributes of `observation`
        are flattened (except along the batch axis):
            - ops_machine_ids
            - ops_durations
            - machines_job_ids
            - machines_remaining_times

        Args:
            observation: `Observation` that the agent sees.

        Returns:
            Concatenation of operations embedding and machines embedding.
        """
        # Flatten, apply operations mask, and obtain operations embedding
        batch_size = observation.machines_job_ids.shape[0]
        machine_ids_flattened = observation.ops_machine_ids.reshape(batch_size, -1)
        durations_flattened = observation.ops_durations.reshape(batch_size, -1)
        ops_mask = observation.ops_mask.reshape(batch_size, -1)
        machine_ids_flattened = jnp.where(ops_mask, machine_ids_flattened, -1)
        durations_flattened = jnp.where(ops_mask, durations_flattened, -1)
        ops_embedding = jnp.concatenate(
            [machine_ids_flattened, durations_flattened], axis=-1, dtype=float
        )
        ops_embedding_layer = hk.nets.MLP(self.operations_layers, activate_final=True)
        ops_embedding = ops_embedding_layer(ops_embedding)

        # Flatten and obtain embedding for status of machines
        batch_shape = observation.machines_job_ids.shape[:-1]
        job_ids_flattened = observation.machines_job_ids.reshape(*batch_shape, -1)
        remaining_times_flattened = observation.machines_remaining_times.reshape(
            *batch_shape, -1
        )
        machines_embedding = jnp.concatenate(
            [job_ids_flattened, remaining_times_flattened], axis=-1, dtype=float
        )
        machines_embedding_layer = hk.nets.MLP(
            self.machines_layers, activate_final=True
        )
        machines_embedding = machines_embedding_layer(machines_embedding)

        # Combine operations embedding and machines embedding
        x = jnp.concatenate([ops_embedding, machines_embedding], axis=-1)

        return x
