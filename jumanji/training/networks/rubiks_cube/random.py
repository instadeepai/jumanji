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
import jax

from jumanji.environments.logic.rubiks_cube import Observation, RubiksCube
from jumanji.training.networks.protocols import RandomPolicy


def make_random_policy_rubiks_cube(rubiks_cube: RubiksCube) -> RandomPolicy:
    """Make random policy for RubiksCube."""
    action_minimum = rubiks_cube.action_spec().minimum
    action_maximum = rubiks_cube.action_spec().maximum

    def random_policy(observation: Observation, key: chex.PRNGKey) -> chex.Array:
        batch_size = observation.cube.shape[0]
        action = jax.random.randint(
            key,
            (batch_size, len(action_minimum)),
            minval=action_minimum,
            maxval=action_maximum,
        )
        return action

    return random_policy
