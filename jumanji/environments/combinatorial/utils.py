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


from chex import Array
import jax.numpy as jnp


def get_augmentations(problem: Array) -> Array:
    """
       Returns the 8 augmentations of a given instance problem described in [1].
       [1] https://arxiv.org/abs/2010.16011
       Args:
           problem: array of coordinates for all cities [problem_size, 2]
       Returns:
           Array with 8 augmentations [8, problem_size, 2]
       """

    # Coordinates -> (1 - coordinates) for each city
    p_aug1 = jnp.array(
        [
            problem,
            jnp.transpose(jnp.array([1 - problem[:, 0], problem[:, 1]])),
            jnp.transpose(jnp.array([problem[:, 0], 1 - problem[:, 1]])),
            jnp.transpose(jnp.array([1 - problem[:, 0], 1 - problem[:, 1]])),
        ]
    )

    # Coordinates are also flipped
    p_aug2 = jnp.einsum("ijk -> jki", jnp.array([p_aug1[:, :, 1], p_aug1[:, :, 0]]))

    return jnp.concatenate([p_aug1, p_aug2], axis=0)
