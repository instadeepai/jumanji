from typing import Dict

import jax.numpy as jnp
import pytest
from jax import random

import jumanji.testing.fakes as fakes
import jumanji.testing.pytrees as pytrees
from jumanji.env import make_environment_spec
from jumanji.types import Action, TimeStep
from validation.agents import TrainingState, Transition

fake_jax_env = pytest.fixture(fakes.make_fake_jax_env)
fake_multi_jax_env = pytest.fixture(fakes.make_fake_multi_jax_env)
fake_dm_env = pytest.fixture(fakes.make_fake_dm_env)
fake_agent = pytest.fixture(fakes.make_fake_agent)


@pytest.mark.parametrize("fake_jax_env", [()], indirect=True)
def test_fake_jax_env__reset(fake_jax_env: fakes.FakeJaxEnv) -> None:
    """Validates the reset of the fake jax environment."""
    state, timestep, _ = fake_jax_env.reset(random.PRNGKey(0))
    assert isinstance(state, fakes.FakeState)
    assert isinstance(timestep, TimeStep)


@pytest.mark.parametrize("fake_jax_env", [()], indirect=True)
def test_fake_jax_env__step(fake_jax_env: fakes.FakeJaxEnv) -> None:
    """Validates the step function of the fake jax environment."""
    state, timestep, _ = fake_jax_env.reset(random.PRNGKey(0))
    action = fake_jax_env.action_spec().generate_value()
    next_state, timestep, _ = fake_jax_env.step(state, action)
    # Check that the step value is now different
    assert state.step != next_state.step


@pytest.mark.parametrize("fake_jax_env", [()], indirect=True)
def test_fake_jax_env__does_not_smoke(fake_jax_env: fakes.FakeJaxEnv) -> None:
    """Validates the run of an episode in the fake jax environment. Check that it does not smoke."""
    state, timestep, _ = fake_jax_env.reset(random.PRNGKey(0))
    action = fake_jax_env.action_spec().generate_value()
    while not timestep.last():
        state, timestep, _ = fake_jax_env.step(state, action)


@pytest.mark.parametrize("fake_multi_jax_env", [()], indirect=True)
def test_fake_multi_jax_env__reset(fake_multi_jax_env: fakes.FakeMultiJaxEnv) -> None:
    """Validates the reset of the fake multi agent jax environment."""
    state, timestep, _ = fake_multi_jax_env.reset(random.PRNGKey(0))
    assert isinstance(state, fakes.FakeState)
    assert isinstance(timestep, TimeStep)
    assert timestep.reward.shape == (fake_multi_jax_env.num_agents,)
    assert timestep.discount.shape == (fake_multi_jax_env.num_agents,)
    assert timestep.observation.shape[0] == fake_multi_jax_env.num_agents


@pytest.mark.parametrize("fake_multi_jax_env", [()], indirect=True)
def test_fake_multi_jax_env__step(fake_multi_jax_env: fakes.FakeMultiJaxEnv) -> None:
    """Validates the step function of the fake multi agent jax environment."""
    state, timestep, _ = fake_multi_jax_env.reset(random.PRNGKey(0))
    action = fake_multi_jax_env.action_spec().generate_value()
    assert action.shape[0] == fake_multi_jax_env.num_agents

    next_state, timestep, _ = fake_multi_jax_env.step(state, action)
    # Check that the step value is now different
    assert state.step != next_state.step
    assert timestep.reward.shape == (fake_multi_jax_env.num_agents,)
    assert timestep.discount.shape == (fake_multi_jax_env.num_agents,)
    assert timestep.observation.shape[0] == fake_multi_jax_env.num_agents


@pytest.mark.parametrize("fake_multi_jax_env", [()], indirect=True)
def test_fake_multi_jax_env__does_not_smoke(
    fake_multi_jax_env: fakes.FakeMultiJaxEnv,
) -> None:
    """Validates the run of an episode in the fake multi agent jax environment. Check that it does not smoke."""
    state, timestep, _ = fake_multi_jax_env.reset(random.PRNGKey(0))
    action = fake_multi_jax_env.action_spec().generate_value()
    assert action.shape[0] == fake_multi_jax_env.num_agents
    while not timestep.last():
        state, timestep, _ = fake_multi_jax_env.step(state, action)


class TestFakeAgent:
    fake_jax_env = fakes.make_fake_jax_env()
    fake_agent = fakes.FakeAgent()

    def test_fake_agent__init_training_state(self) -> TrainingState:
        """Test and return the training state."""
        key = random.PRNGKey(0)
        training_state = self.fake_agent.init_training_state(key)
        return training_state

    def test_fake_agent__select_action(self) -> None:
        """Check the dummy action selection."""
        training_state = self.test_fake_agent__init_training_state()
        observation = jnp.zeros(())
        key = random.PRNGKey(1)
        action = self.fake_agent.select_action(training_state, observation, key)
        assert isinstance(action, Action)

    def test_fake_agent__sgd_step(self) -> None:
        """Check the dummy sgd step."""
        training_state = self.test_fake_agent__init_training_state()
        batch_traj = fakes.batch_fake_traj(make_environment_spec(self.fake_jax_env))
        new_training_state, metrics = self.fake_agent.sgd_step(
            training_state, batch_traj
        )
        assert isinstance(metrics, Dict)
        assert isinstance(training_state, TrainingState)
        assert isinstance(new_training_state, TrainingState)


class TestFakeMultiAgent:
    fake_multi_jax_env = fakes.make_fake_multi_jax_env()
    fake_multi_agent = fakes.FakeMultiAgent(
        fake_multi_jax_env.action_spec(), fake_multi_jax_env.num_agents
    )

    def test_fake_multi_agent__init_training_state(self) -> TrainingState:
        """Test and return the training state."""
        key = random.PRNGKey(0)
        training_state = self.fake_multi_agent.init_training_state(key)
        return training_state

    def test_fake_multi_agent__select_action(self) -> None:
        """Check the dummy action selection."""
        training_state = self.test_fake_multi_agent__init_training_state()
        observation = jnp.zeros(self.fake_multi_jax_env.observation_shape, float)
        key = random.PRNGKey(1)
        action = self.fake_multi_agent.select_action(training_state, observation, key)
        assert isinstance(action, Action)
        assert action.shape[0] == self.fake_multi_jax_env.num_agents


@pytest.mark.parametrize("fake_jax_env", [()], indirect=True)
def test_fake_transition(fake_jax_env: fakes.FakeJaxEnv) -> None:
    """Check that the fake transition has one attribute (e.g. reward) that is of rank 0."""
    env_spec = make_environment_spec(fake_jax_env)
    fake_transition = fakes.fake_transition(env_spec)
    assert isinstance(fake_transition, Transition)
    pytrees.assert_is_jax_array_tree(fake_transition)
    assert pytrees.has_at_least_rank(fake_transition, 0)
    assert not pytrees.has_at_least_rank(fake_transition, 1)


@pytest.mark.parametrize("fake_jax_env", [()], indirect=True)
def test_fake_traj(fake_jax_env: fakes.FakeJaxEnv) -> None:
    """Check that the fake trajectory has all its attributes with a leading dimension, and
    one attribute (e.g. reward) that is of rank 1."""
    env_spec = make_environment_spec(fake_jax_env)
    n_steps = 3
    fake_traj = fakes.fake_traj(env_spec, n_steps)
    assert isinstance(fake_traj, Transition)
    pytrees.assert_is_jax_array_tree(fake_traj)
    assert pytrees.has_at_least_rank(fake_traj, 1)
    assert not pytrees.has_at_least_rank(fake_traj, 2)


@pytest.mark.parametrize("fake_jax_env", [()], indirect=True)
def test_batch_fake_traj(fake_jax_env: fakes.FakeJaxEnv) -> None:
    """Check that the fake batch of trajectories has all its attributes with two leading dimensions,
    and one attribute (e.g. reward) that is of rank 2."""
    env_spec = make_environment_spec(fake_jax_env)
    n_steps = 3
    batch_size = 2
    fake_batch_traj = fakes.batch_fake_traj(
        env_spec, n_steps=n_steps, batch_size=batch_size
    )
    assert isinstance(fake_batch_traj, Transition)
    pytrees.assert_is_jax_array_tree(fake_batch_traj)
    assert pytrees.has_at_least_rank(fake_batch_traj, 2)
    assert not pytrees.has_at_least_rank(fake_batch_traj, 3)
