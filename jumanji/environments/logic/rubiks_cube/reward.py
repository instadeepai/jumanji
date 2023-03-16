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

import chex
import jax.numpy as jnp

from jumanji.environments.logic.rubiks_cube.types import State
from jumanji.environments.logic.rubiks_cube.utils import is_solved


class RewardFn(abc.ABC):
    @abc.abstractmethod
    def __call__(self, state: State) -> chex.Array:
        """Call method for computing the reward given new state."""


class SparseRewardFn(RewardFn):
    """A sparse reward function, returning +1 if cube is solved and otherwise 0."""

    def __call__(self, state: State) -> chex.Array:
        solved = is_solved(state.cube)
        return jnp.array(solved, float)
