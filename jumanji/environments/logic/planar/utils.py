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

import jax
from chex import Array
from jax import numpy as jnp


def check_intersect(
    segment1: Array, segment2: Array, edge1: Array, edge2: Array
) -> jnp.bool_:
    # Cond 1: intersect if the 2 segments share a common endpoint,
    # except if they correspond to a shared node
    nodes_comp = jnp.equal(
        jnp.expand_dims(edge1, axis=-1), jnp.expand_dims(edge2, axis=0)
    )
    position_comp = jnp.equal(
        jnp.expand_dims(segment1, axis=-2), jnp.expand_dims(segment2, axis=-3)
    ).all(axis=-1)

    cond1 = jnp.any(jnp.not_equal(nodes_comp, position_comp))

    # Cond 2: Do not intersect if the segments are parallel
    # build the linear system
    (p1, p2), (q1, q2) = segment1, segment2
    (a, c), (b, d), (bx, by) = p2 - p1, q1 - q2, q1 - p1

    det = a * d - b * c
    cond2 = jnp.not_equal(det, 0)

    # Cond 3: intersect if there exists a solution of the system Ax=b such that 0 <= x <= 1.
    # solve the inverse problem
    alpha = (d * bx - b * by) / det
    beta = (a * by - c * bx) / det

    x_cross = jnp.logical_and(jnp.greater(alpha, 0), jnp.less(alpha, 1))
    y_cross = jnp.logical_and(jnp.greater(beta, 0), jnp.less(beta, 1))
    cond3 = jnp.logical_and(x_cross, y_cross)

    return jnp.logical_or(cond1, jnp.logical_and(cond2, cond3))


def intersection_map(segments: Array, edges: Array) -> Array:
    # All-to-all comparison between segments
    intersect_fn = jax.vmap(check_intersect, in_axes=(0, None, 0, None))
    intersect_fn = jax.vmap(intersect_fn, in_axes=(None, 0, None, 0))
    intersect = intersect_fn(segments, segments, edges, edges)
    return intersect


def intersection_count(segments: Array, edges: Array) -> jnp.int_:
    # Count the effective number of crossings
    intersect = intersection_map(segments, edges)
    num_intersections = jnp.sum(intersect)
    num_intersections = num_intersections // 2  # each crossing is counted twice

    return num_intersections
