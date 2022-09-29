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

from chex import Array, PRNGKey
from jax import numpy as jnp
from jax import random
from jumanji.environments.combinatorial.tsp.utils import get_augmentations as get_augmentations_tsp

DEPOT_IDX = 0
MIN_NORM_FACTOR = 10


def compute_tour_length(problem: Array, order: Array) -> jnp.float32:
    """Calculate the length of a tour."""
    problem = problem[order, 0:2]
    return jnp.linalg.norm((problem - jnp.roll(problem, -1, axis=0)), axis=1).sum()


def generate_problem(problem_key: PRNGKey, num_nodes: jnp.int32) -> Array:
    coords = random.uniform(problem_key, (num_nodes + 1, 2), minval=0, maxval=1)
    costs = random.randint(problem_key, (num_nodes + 1, 1), minval=1, maxval=10)
    problem = jnp.hstack((coords, costs))
    problem = problem.at[DEPOT_IDX, 2].set(0.0)
    return problem


def generate_start_position(start_key: PRNGKey, num_nodes: jnp.int32) -> jnp.int32:
    return random.randint(start_key, (), minval=1, maxval=num_nodes + 1)


def get_augmentations(problem: Array) -> Array:
    """
    Returns the 8 augmentations of a given instance problem described in [1]. This function leverages the existing
    augmentation method for TSP and appends the costs/demands used in CVRP.
    [1] https://arxiv.org/abs/2010.16011

    Args:
        problem: Array of coordinates and demands/costs for all nodes [num_nodes, 3]

    Returns:
        augmentations: Array with 8 augmentations [8, num_nodes, 3]
    """
    coord_augmentations = get_augmentations_tsp(problem[:, :2])

    num_nodes = problem.shape[0]
    num_augmentations = coord_augmentations.shape[0]

    costs_per_aug = jnp.tile(problem[:, 2], num_augmentations).reshape(
        num_augmentations, num_nodes, 1
    )
    return jnp.concatenate((coord_augmentations, costs_per_aug), axis=2)
