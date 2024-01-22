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

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class State:
    """
    key: random key used for auto-reset.
    fixed_grid: Array (uint8) shape (n_rows, n_cols) array representing the
    fixed elements of a sokoban problem.
    variable_grid: Array (uint8) shape (n_rows, n_cols) array representing the
    variable elements of a sokoban problem.
    agent_location: Array (int32) shape (2,)
    step_count: Array (int32) shape ()
    """

    key: chex.PRNGKey
    fixed_grid: chex.Array
    variable_grid: chex.Array
    agent_location: chex.Array
    step_count: chex.Array


class Observation(NamedTuple):
    """
    The observation returned by the sokoban environment.
    grid: Array (uint8) shape (n_rows, n_cols, 2) array representing the
    variable and fixed grids.
    step_count: Array (int32) shape () the index of the current step.
    """

    grid: chex.Array
    step_count: chex.Array
