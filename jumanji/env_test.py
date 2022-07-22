from typing import Type

import pytest
import pytest_mock

from jumanji.env import Wrapper
from jumanji.testing.fakes import FakeEnvironment, FakeState, make_fake_environment


@pytest.fixture
def mock_wrapper_class() -> Type[Wrapper]:
    class MockWrapper(Wrapper[FakeState]):
        pass

    return MockWrapper


@pytest.fixture
def mock_env() -> FakeEnvironment:
    return make_fake_environment()


@pytest.fixture
def wrapped_mock_env(
    mock_wrapper_class: Type[Wrapper], mock_env: FakeEnvironment
) -> Wrapper:
    wrapped_env = mock_wrapper_class(mock_env)
    return wrapped_env


def test_wrapper__unwrapped(
    wrapped_mock_env: Wrapper, mock_env: FakeEnvironment
) -> None:
    """Checks Wrapper.unwrapped returns the unwrapped env"""
    assert wrapped_mock_env.unwrapped == mock_env


def test_wrapper__step(
    mocker: pytest_mock.MockerFixture,
    wrapped_mock_env: Wrapper,
    mock_env: FakeEnvironment,
) -> None:
    """Checks Wrapper.step calls the step method of the underlying env"""
    mock_step = mocker.patch.object(mock_env, "step", autospec=True)
    mock_state = mocker.MagicMock()
    mock_action = mocker.MagicMock()

    wrapped_mock_env.step(mock_state, mock_action)

    mock_step.assert_called_once_with(mock_state, mock_action)


def test_wrapper__reset(
    mocker: pytest_mock.MockerFixture,
    wrapped_mock_env: Wrapper,
    mock_env: FakeEnvironment,
) -> None:
    """Checks Wrapper.reset calls the reset method of the underlying env"""
    mock_reset = mocker.patch.object(mock_env, "reset", autospec=True)
    mock_key = mocker.MagicMock()

    wrapped_mock_env.reset(mock_key)

    mock_reset.assert_called_once_with(mock_key)


def test_wrapper__observation_spec(
    mocker: pytest_mock.MockerFixture,
    wrapped_mock_env: Wrapper,
    mock_env: FakeEnvironment,
) -> None:
    """Checks Wrapper.observation_spec calls the observation_spec function of the underlying env"""
    mock_obs_spec = mocker.patch.object(mock_env, "observation_spec", autospec=True)

    wrapped_mock_env.observation_spec()

    mock_obs_spec.assert_called_once()


def test_wrapper__action_spec(
    mocker: pytest_mock.MockerFixture,
    wrapped_mock_env: Wrapper,
    mock_env: FakeEnvironment,
) -> None:
    """Checks Wrapper.action_spec calls the action_spec function of the underlying env"""
    mock_action_spec = mocker.patch.object(mock_env, "action_spec", autospec=True)

    wrapped_mock_env.action_spec()

    mock_action_spec.assert_called_once()


def test_wrapper__repr(wrapped_mock_env: Wrapper, mock_env: FakeEnvironment) -> None:
    """Checks Wrapper.__repr__ returns the expected representation string"""
    repr_str = repr(wrapped_mock_env)
    assert "MockWrapper" in repr_str


def test_wrapper__getattr(wrapped_mock_env: Wrapper, mock_env: FakeEnvironment) -> None:
    """Checks Wrapper.__getattr__ calls the underlying env for unknown attr."""
    # time_limit is defined in the mock env
    assert wrapped_mock_env.time_limit == mock_env.time_limit
