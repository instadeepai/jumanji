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
from validation.benchmark_loops import DeepMindEnvBenchmarkLoop


@pytest.mark.parametrize("fake_dm_env", [()], indirect=True)
def test_deep_mind_env_benchmark_loop__init(fake_dm_env: dm_env.Environment) -> None:
    """Validates initialization of the dm_env benchmark loop."""
    deep_mind_env_benchmark_loop = DeepMindEnvBenchmarkLoop(fake_dm_env)
    assert isinstance(deep_mind_env_benchmark_loop, DeepMindEnvBenchmarkLoop)


@pytest.mark.parametrize("fake_environment", [()], indirect=True)
def test_deep_mind_env_benchmark_loop__init_env_check(
    fake_environment: FakeEnvironment,
) -> None:
    """Validates that the environment loop raises an issue if the environment is not a
    dm_env.Environment."""
    gym_env = gym.make("CartPole-v0")
    with pytest.raises(TypeError):
        DeepMindEnvBenchmarkLoop(gym_env)
    with pytest.raises(TypeError):
        DeepMindEnvBenchmarkLoop(fake_environment)


@pytest.mark.parametrize("fake_dm_env", [()], indirect=True)
def test_dm_env_benchmark__run(
    fake_dm_env: dm_env.Environment, capsys: pytest.CaptureFixture
) -> None:
    """Validates dm_env benchmark loop run method on a fake environment."""
    deep_mind_env_benchmark_loop = DeepMindEnvBenchmarkLoop(fake_dm_env)
    deep_mind_env_benchmark_loop.run(num_episodes=1)
    assert capsys.readouterr().out
