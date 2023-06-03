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

from jumanji.environments.packing.tetris import Tetris
from jumanji.training.networks.masked_categorical_random import (
    make_masked_categorical_random_ndim,
)
from jumanji.training.networks.protocols import RandomPolicy


def make_random_policy_tetris(tetris: Tetris) -> RandomPolicy:
    """Make random policy for `Tetris`."""
    action_spec_num_values = tetris.action_spec().num_values
    return make_masked_categorical_random_ndim(
        action_spec_num_values=action_spec_num_values
    )
