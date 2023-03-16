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

from typing import TYPE_CHECKING, NamedTuple

import chex
import jax.random
from chex import Array
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

Cube: TypeAlias = Array


@dataclass
class State:
    """
    cube: 3D array whose cells contain the index of the corresponding colour of the sticker in the
        scramble.
    step_count: specifies how many timesteps have elapsed since environment reset.
    action_history: array that indicates the entire history of applied moves (including those taken
        on scrambling the cube in the environment reset).
    key: random key used for seeding the sampling for scrambling on reset.
    """

    cube: Cube  # (6, cube_size, cube_size)
    step_count: chex.Numeric  # ()
    action_history: Array  # (num_scrambles_on_reset + time_limit, 3)
    key: chex.PRNGKey = jax.random.PRNGKey(0)  # (2,)


class Observation(NamedTuple):
    """
    cube: 3D array whose cells contain the index of the corresponding colour of the sticker in the
        scramble.
    step_count: specifies how many timesteps have elapsed since environment reset.
    """

    cube: Cube  # (6, cube_size, cube_size)
    step_count: chex.Numeric  # ()
