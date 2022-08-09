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

from jumanji.testing.fakes import FakeEnvironment
from validation.benchmark_loops import EnvironmentBenchmarkLoop


@pytest.mark.parametrize("fake_environment", [()], indirect=True)
def test_environment_benchmark_loop__init(fake_environment: FakeEnvironment) -> None:
    """Validates initialization of the jax benchmark loop."""
    environment_benchmark_loop = EnvironmentBenchmarkLoop(fake_environment)
    assert isinstance(environment_benchmark_loop, EnvironmentBenchmarkLoop)


@pytest.mark.parametrize("fake_dm_env", [()], indirect=True)
def test_environment_benchmark_loop__init_env_check(
    fake_dm_env: dm_env.Environment,
) -> None:
    """Validates that the environment loop raises an issue if the environment is not an Environment."""
    gym_env = gym.make("CartPole-v0")
    with pytest.raises(TypeError):
        EnvironmentBenchmarkLoop(gym_env)
    with pytest.raises(TypeError):
        EnvironmentBenchmarkLoop(fake_dm_env)


@pytest.mark.parametrize("fake_environment", [()], indirect=True)
def test_environment_benchmark_run__does_not_smoke(
    fake_environment: FakeEnvironment, capsys: pytest.CaptureFixture
) -> None:
    """Validates jax benchmark loop run method on a fake environment."""
    environment_benchmark_loop = EnvironmentBenchmarkLoop(fake_environment)
    environment_benchmark_loop.run(num_episodes=1)
    assert capsys.readouterr().out
