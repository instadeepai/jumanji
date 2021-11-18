import dm_env
import gym
import pytest

from jumanji.testing.fakes import FakeJaxEnv
from jumanji.utils.jax_env_benchmark_loop import JaxEnvBenchmarkLoop


@pytest.mark.parametrize("fake_jax_env", [()], indirect=True)
def test_jax_env_benchmark_loop__init(fake_jax_env: FakeJaxEnv) -> None:
    """Validates initialization of the jax benchmark loop."""
    jax_env_benchmark_loop = JaxEnvBenchmarkLoop(fake_jax_env)
    assert isinstance(jax_env_benchmark_loop, JaxEnvBenchmarkLoop)


@pytest.mark.parametrize("fake_dm_env", [()], indirect=True)
def test_jax_env_benchmark_loop__init_env_check(
    fake_dm_env: dm_env.Environment,
) -> None:
    """Validates that the environment loop raises an issue if the environment is not a JaxEnv."""
    gym_env = gym.make("CartPole-v0")
    with pytest.raises(TypeError):
        JaxEnvBenchmarkLoop(gym_env)
    with pytest.raises(TypeError):
        JaxEnvBenchmarkLoop(fake_dm_env)


@pytest.mark.parametrize("fake_jax_env", [()], indirect=True)
def test_jax_env_benchmark_run__does_not_smoke(fake_jax_env: FakeJaxEnv) -> None:
    """Validates jax benchmark loop run method on a fake environment."""
    jax_env_benchmark_loop = JaxEnvBenchmarkLoop(fake_jax_env)
    jax_env_benchmark_loop.run(num_episodes=1, print_=False)
