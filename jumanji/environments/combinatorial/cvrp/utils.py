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

from typing import Tuple

import jax
from chex import Array, PRNGKey
from jax import numpy as jnp

from jumanji.environments.combinatorial.utils import get_coordinates_augmentations

DEPOT_IDX = 0


def compute_tour_length(coordinates: Array, order: Array) -> jnp.float32:
    """Calculate the length of a tour."""
    coordinates = coordinates[order]
    return jnp.linalg.norm(
        (coordinates - jnp.roll(coordinates, -1, axis=0)), axis=1
    ).sum()


def generate_problem(
    key: PRNGKey, num_nodes: jnp.int32, max_demand: jnp.int32
) -> Tuple[Array, Array]:
    coord_key, demand_key = jax.random.split(key)
    coords = jax.random.uniform(coord_key, (num_nodes + 1, 2), minval=0, maxval=1)
    demands = jax.random.randint(
        demand_key, (num_nodes + 1,), minval=1, maxval=max_demand
    )
    demands = demands.at[DEPOT_IDX].set(0)
    return coords, demands


def generate_start_position(key: PRNGKey, num_nodes: jnp.int32) -> jnp.int32:
    return jax.random.randint(key, (), minval=1, maxval=num_nodes + 1)


def get_augmentations(coordinates: Array, demands: Array) -> Tuple[Array, Array]:
    """Returns the 8 augmentations of a given instance problem described in [1]. This function
    leverages the existing augmentation method for TSP and appends the costs/demands used in CVRP.
    [1] https://arxiv.org/abs/2010.16011

    Args:
        coordinates: Array of coordinates for all nodes [num_nodes, 2]
        demands: Array of demands for all nodes [num_nodes]

    Returns:
        coord_augmentations: Array with 8 coordinates augmentations [8, num_nodes, 2]
        demands_augmentations: Array with 8 demands augmentations [8, num_nodes]
    """
    coord_augmentations = get_coordinates_augmentations(coordinates)

    num_augmentations = coord_augmentations.shape[0]

    demands_augmentations = jnp.tile(demands, (num_augmentations, 1))

    return coord_augmentations, demands_augmentations
