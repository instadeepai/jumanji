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
import jax.random
import pytest

from jumanji.environments.swarms.search_and_rescue import SearchAndRescue, observations


@pytest.fixture
def env() -> SearchAndRescue:
    observation_fn = observations.AgentAndTargetObservationFn(
        num_vision=32,
        searcher_vision_range=0.1,
        target_vision_range=0.1,
        view_angle=0.5,
        agent_radius=0.01,
        env_size=1.0,
    )
    return SearchAndRescue(
        target_contact_range=0.05,
        searcher_max_rotate=0.2,
        searcher_max_accelerate=0.01,
        searcher_min_speed=0.01,
        searcher_max_speed=0.05,
        searcher_view_angle=0.5,
        time_limit=10,
        observation=observation_fn,
    )


@pytest.fixture
def key() -> chex.PRNGKey:
    return jax.random.PRNGKey(101)
