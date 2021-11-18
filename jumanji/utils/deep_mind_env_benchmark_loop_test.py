import dm_env
import gym
import pytest

from jumanji.testing.fakes import FakeJaxEnv
from jumanji.utils.deep_mind_env_benchmark_loop import DeepMindEnvBenchmarkLoop


@pytest.mark.parametrize("fake_dm_env", [()], indirect=True)
def test_deep_mind_env_benchmark_loop__init(fake_dm_env: dm_env.Environment) -> None:
    """Validates initialization of the dm_env benchmark loop."""
    deep_mind_env_benchmark_loop = DeepMindEnvBenchmarkLoop(fake_dm_env)
    assert isinstance(deep_mind_env_benchmark_loop, DeepMindEnvBenchmarkLoop)


@pytest.mark.parametrize("fake_jax_env", [()], indirect=True)
def test_deep_mind_env_benchmark_loop__init_env_check(fake_jax_env: FakeJaxEnv) -> None:
    """Validates that the environment loop raises an issue if the environment is not a
    dm_env.Environment."""
    gym_env = gym.make("CartPole-v0")
    with pytest.raises(TypeError):
        DeepMindEnvBenchmarkLoop(gym_env)
    with pytest.raises(TypeError):
        DeepMindEnvBenchmarkLoop(fake_jax_env)


@pytest.mark.parametrize("fake_dm_env", [()], indirect=True)
def test_dm_env_benchmark__run(fake_dm_env: dm_env.Environment) -> None:
    """Validates dm_env benchmark loop run method on a fake environment."""
    deep_mind_env_benchmark_loop = DeepMindEnvBenchmarkLoop(fake_dm_env)
    deep_mind_env_benchmark_loop.run(num_episodes=1, print_=False)
