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

from jumanji.environments.routing.routing import Routing
from jumanji.training.networks.random_policy import RandomPolicy


def make_random_policy_routing(routing: Routing) -> RandomPolicy:
    """Make random policy for Routing. The action mask is not given, so we may sample actions that
    lead to a terminal state.
    """
    minval = routing.action_spec().minimum[0]
    maxval = routing.action_spec().maximum[0]

    def random_policy(observation: chex.Array, key: chex.PRNGKey) -> chex.Array:
        batch_shape = observation.shape[:-3]
        action = jax.random.randint(
            key,
            (*batch_shape, routing.num_agents),
            minval=minval,
            maxval=maxval,
        )
        return action

    return random_policy
