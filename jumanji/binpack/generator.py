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

import abc

from chex import PRNGKey

from jumanji.binpack.types import State

# 20 foot container: 5.898m long x 2.352m wide x 2.393m high
# Oftentimes people use different values (inner volume) : 5.870m long x 2.330m wide x 2.200m high
# real_container_volume = 1.103 * inner_container_volume
TWENTY_FOOT_DIMS = (5870, 2330, 2200)


class Generator(abc.ABC):
    """
    Defines the abstract Generator base class.
    """

    def __init__(self, max_num_items: int, max_num_ems: int):
        """Abstract class implementing `max_num_items` and `max_num_ems` properties.

        Args:
            max_num_items: maximum number of items the generator will ever generate when creating
                a new instance. This defines the shapes of arrays related to items in the
                environment state.
            max_num_ems: maximum number of ems the environment will handle. This defines the shape
                of the EMS buffer that is kept in the environment state.
        """
        self._max_num_items = max_num_items
        self._max_num_ems = max_num_ems

    @property
    def max_num_items(self) -> int:
        """Returns the maximum number of items the generator will ever create when resetting the
        environment."""
        return self._max_num_items

    @property
    def max_num_ems(self) -> int:
        """Returns the maximum number of ems the environment will handle. Any created ems that do
        not fit in the buffer will be ignored during the environment step."""
        return self._max_num_ems

    @abc.abstractmethod
    def __call__(self, key: PRNGKey) -> State:
        """Call method responsible for generating a new state.

        Args:
            key: jax random key for any stochasticity used in the generation process

        Returns:
            A BinPack State.
        """
