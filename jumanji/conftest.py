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
import os

import dm_env
import pytest

from jumanji.testing.fakes import FakeEnvironment, FakeMultiEnvironment
from jumanji.wrappers import JumanjiToDMEnvWrapper

# Use headless driver for Pygame.
os.environ["SDL_VIDEODRIVER"] = "dummy"


@pytest.fixture
def fake_environment(time_limit: int = 10) -> FakeEnvironment:
    """Creates a fake environment."""
    return FakeEnvironment(time_limit=time_limit)


@pytest.fixture
def fake_multi_environment(time_limit: int = 10) -> FakeMultiEnvironment:
    """Creates a fake multi agent environment."""
    return FakeMultiEnvironment(time_limit=time_limit)


@pytest.fixture
def fake_dm_env(time_limit: int = 10) -> dm_env.Environment:
    """Creates a fake environment wrapped as a dm_env.Environment."""
    return JumanjiToDMEnvWrapper(FakeEnvironment(time_limit=time_limit))
