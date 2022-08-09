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

import dm_env
import gym
import pytest

from jumanji import Environment
from jumanji.testing.fakes import FakeAgent, FakeEnvironment
from validation.agents import Agent
from validation.environment_loops.environment_loop import EnvironmentLoop


@pytest.mark.parametrize("fake_environment", [()], indirect=True)
@pytest.mark.parametrize("fake_agent", [()], indirect=True)
def test_environment_loop__init(
    fake_environment: FakeEnvironment, fake_agent: FakeAgent
) -> None:
    """Validates initialization of the environment loop."""
    environment_loop = EnvironmentLoop(fake_environment, fake_agent)
    assert isinstance(environment_loop, EnvironmentLoop)
    assert isinstance(environment_loop._agent, Agent)
    assert isinstance(environment_loop._environment, Environment)


@pytest.mark.parametrize("fake_dm_env", [()], indirect=True)
@pytest.mark.parametrize("fake_agent", [()], indirect=True)
def test_environment_benchmark_loop__init_env_check(
    fake_dm_env: dm_env.Environment,
    fake_agent: FakeAgent,
) -> None:
    """Validates that the environment loop raises an issue if the environment is not an Environment."""
    gym_env = gym.make("CartPole-v0")
    with pytest.raises(TypeError):
        EnvironmentLoop(gym_env, fake_agent)
    with pytest.raises(TypeError):
        EnvironmentLoop(fake_dm_env, fake_agent)


@pytest.mark.parametrize("fake_environment", [()], indirect=True)
@pytest.mark.parametrize("fake_agent", [()], indirect=True)
def test_environment_loop__run(
    fake_environment: FakeEnvironment,
    fake_agent: FakeAgent,
    capsys: pytest.CaptureFixture,
) -> None:
    """Validates environment loop run method on a fake environment."""
    environment_loop = EnvironmentLoop(
        fake_environment, fake_agent, n_steps=1, batch_size=1
    )
    environment_loop.run(num_episodes=1)
    environment_loop = EnvironmentLoop(
        fake_environment, fake_agent, n_steps=3, batch_size=2
    )
    environment_loop.run(num_steps=5)
    assert capsys.readouterr().out
