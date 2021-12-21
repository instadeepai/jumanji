import dm_env
import gym
import pytest

from jumanji.jax import JaxEnv
from jumanji.testing.fakes import FakeAgent, FakeJaxEnv
from validation.agents import Agent
from validation.environment_loops.jax_environment_loop import JaxEnvironmentLoop


@pytest.mark.parametrize("fake_jax_env", [()], indirect=True)
@pytest.mark.parametrize("fake_agent", [()], indirect=True)
def test_jax_environment_loop__init(
    fake_jax_env: FakeJaxEnv, fake_agent: FakeAgent
) -> None:
    """Validates initialization of the environment loop."""
    jax_environment_loop = JaxEnvironmentLoop(fake_jax_env, fake_agent)
    assert isinstance(jax_environment_loop, JaxEnvironmentLoop)
    assert isinstance(jax_environment_loop._agent, Agent)
    assert isinstance(jax_environment_loop._environment, JaxEnv)


@pytest.mark.parametrize("fake_dm_env", [()], indirect=True)
@pytest.mark.parametrize("fake_agent", [()], indirect=True)
def test_jax_env_benchmark_loop__init_env_check(
    fake_dm_env: dm_env.Environment,
    fake_agent: FakeAgent,
) -> None:
    """Validates that the environment loop raises an issue if the environment is not a JaxEnv."""
    gym_env = gym.make("CartPole-v0")
    with pytest.raises(TypeError):
        JaxEnvironmentLoop(gym_env, fake_agent)
    with pytest.raises(TypeError):
        JaxEnvironmentLoop(fake_dm_env, fake_agent)


@pytest.mark.parametrize("fake_jax_env", [()], indirect=True)
@pytest.mark.parametrize("fake_agent", [()], indirect=True)
def test_jax_environment_loop__run(
    fake_jax_env: FakeJaxEnv,
    fake_agent: FakeAgent,
    capsys: pytest.CaptureFixture,
) -> None:
    """Validates jax environment loop run method on a fake environment."""
    jax_environment_loop = JaxEnvironmentLoop(
        fake_jax_env, fake_agent, n_steps=1, batch_size=1
    )
    jax_environment_loop.run(num_episodes=1)
    jax_environment_loop = JaxEnvironmentLoop(
        fake_jax_env, fake_agent, n_steps=3, batch_size=2
    )
    jax_environment_loop.run(num_steps=5)
    assert capsys.readouterr().out
