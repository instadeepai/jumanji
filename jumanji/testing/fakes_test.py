import pytest
from jax import random

from jumanji.jax.env import JaxEnv
from jumanji.jax.types import TimeStep
from jumanji.testing.fakes import (
    FakeJaxEnv,
    FakeState,
    make_fake_dm_env,
    make_fake_jax_env,
)

fake_jax_env = pytest.fixture(make_fake_jax_env)
fake_dm_env = pytest.fixture(make_fake_dm_env)


def test_fake_jax_env__init() -> None:
    """Validates initialization of the fake jax environment."""
    fake_jax_env = FakeJaxEnv()
    assert isinstance(fake_jax_env, JaxEnv)


@pytest.mark.parametrize("fake_jax_env", [()], indirect=True)
def test_fake_jax_env__reset(fake_jax_env: FakeJaxEnv) -> None:
    """Validates the reset of the fake jax environment."""
    state, timestep = fake_jax_env.reset(random.PRNGKey(0))
    assert isinstance(state, FakeState)
    assert isinstance(timestep, TimeStep)


@pytest.mark.parametrize("fake_jax_env", [()], indirect=True)
def test_fake_jax_env__step(fake_jax_env: FakeJaxEnv) -> None:
    """Validates the step function of the fake jax environment."""
    state, timestep = fake_jax_env.reset(random.PRNGKey(0))
    action = fake_jax_env.action_spec().generate_value()
    next_state, timestep = fake_jax_env.step(state, action)
    # Check that the step value is now different
    assert state.step != next_state.step


@pytest.mark.parametrize("fake_jax_env", [()], indirect=True)
def test_fake_jax_env__does_not_smoke(fake_jax_env: FakeJaxEnv) -> None:
    """Validates the run of an episode in the fake jax environment. Check that it does not smoke."""
    state, timestep = fake_jax_env.reset(random.PRNGKey(0))
    action = fake_jax_env.action_spec().generate_value()
    while not timestep.last():
        state, timestep = fake_jax_env.step(state, action)
