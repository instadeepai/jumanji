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

import chex
import jax.numpy as jnp

from jumanji.training.networks.distribution import CategoricalDistribution
from jumanji.training.networks.parametric_distribution import ParametricDistribution
from jumanji.training.networks.postprocessor import Postprocessor


class BinPackReshapeBijector(Postprocessor):
    """Identity Bijector that reshapes the output for BinPack."""

    def __init__(self, num_items: int):
        super().__init__()
        self.num_items = num_items

    def forward(self, x: chex.Array) -> chex.Array:
        ems_id, item_id = jnp.divmod(x, self.num_items)
        return jnp.stack([ems_id, item_id], axis=-1)

    def inverse(self, y: chex.Array) -> chex.Array:
        ems_id, item_id = jnp.split(y, axis=-1)
        return ems_id * self.num_items + item_id

    def forward_log_det_jacobian(self, x: chex.Array) -> chex.Array:
        return jnp.zeros_like(x, x.dtype)


class BinPackParametricDistribution(ParametricDistribution):
    """Categorical distribution for BinPack's discrete action space."""

    def __init__(self, num_actions: int, max_num_items: int):
        """Initialize the distribution.
        Args:
            num_actions: the number of actions.
            max_num_items: number of items for BinPack.
        """
        postprocessor = BinPackReshapeBijector(num_items=max_num_items)
        super().__init__(
            param_size=num_actions,
            postprocessor=postprocessor,
            event_ndims=0,
        )

    def create_dist(self, parameters: chex.Array) -> CategoricalDistribution:
        return CategoricalDistribution(logits=parameters)
