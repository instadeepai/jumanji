import dm_env
import gym
import pytest

from jumanji.testing.fakes import FakeJaxEnv
from jumanji.utils.jax_environment_loop import JaxEnvironmentLoop


@pytest.mark.parametrize("fake_jax_env", [()], indirect=True)
def test_jax_environment_loop__init(fake_jax_env: FakeJaxEnv) -> None:
    """Validates initialization of the environment loop."""
    jax_environment_loop = JaxEnvironmentLoop(fake_jax_env)
    assert isinstance(jax_environment_loop, JaxEnvironmentLoop)


@pytest.mark.parametrize("fake_dm_env", [()], indirect=True)
def test_jax_env_benchmark_loop__init_env_check(
    fake_dm_env: dm_env.Environment,
) -> None:
    """Validates that the environment loop raises an issue if the environment is not a JaxEnv."""
    gym_env = gym.make("CartPole-v0")
    with pytest.raises(TypeError):
        JaxEnvironmentLoop(gym_env)
    with pytest.raises(TypeError):
        JaxEnvironmentLoop(fake_dm_env)


@pytest.mark.parametrize("fake_jax_env", [()], indirect=True)
def test_jax_environment_loop__run(fake_jax_env: FakeJaxEnv) -> None:
    """Validates jax environment loop run method on a fake environment."""
    jax_environment_loop = JaxEnvironmentLoop(fake_jax_env, n_steps=1, batch_size=1)
    jax_environment_loop.run(num_episodes=1, print_=False)
    jax_environment_loop = JaxEnvironmentLoop(fake_jax_env, n_steps=3, batch_size=2)
    jax_environment_loop.run(num_episodes=1, print_=False)
