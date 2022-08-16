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

from typing import Type

import pytest
import pytest_mock

from jumanji.env import Wrapper
from jumanji.testing.fakes import FakeEnvironment, FakeState


@pytest.fixture
def mock_wrapper_class() -> Type[Wrapper]:
    class MockWrapper(Wrapper[FakeState]):
        pass

    return MockWrapper


@pytest.fixture
def wrapped_fake_environment(
    mock_wrapper_class: Type[Wrapper], fake_environment: FakeEnvironment
) -> Wrapper:
    wrapped_env = mock_wrapper_class(fake_environment)
    return wrapped_env


def test_wrapper__unwrapped(
    wrapped_fake_environment: Wrapper, fake_environment: FakeEnvironment
) -> None:
    """Checks `Wrapper.unwrapped` returns the unwrapped env."""
    assert wrapped_fake_environment.unwrapped == fake_environment


def test_wrapper__step(
    mocker: pytest_mock.MockerFixture,
    wrapped_fake_environment: Wrapper,
    fake_environment: FakeEnvironment,
) -> None:
    """Checks `Wrapper.step` calls the step method of the underlying env."""
    mock_step = mocker.patch.object(fake_environment, "step", autospec=True)
    mock_state = mocker.MagicMock()
    mock_action = mocker.MagicMock()

    wrapped_fake_environment.step(mock_state, mock_action)

    mock_step.assert_called_once_with(mock_state, mock_action)


def test_wrapper__reset(
    mocker: pytest_mock.MockerFixture,
    wrapped_fake_environment: Wrapper,
    fake_environment: FakeEnvironment,
) -> None:
    """Checks `Wrapper.reset` calls the reset method of the underlying env."""
    mock_reset = mocker.patch.object(fake_environment, "reset", autospec=True)
    mock_key = mocker.MagicMock()

    wrapped_fake_environment.reset(mock_key)

    mock_reset.assert_called_once_with(mock_key)


def test_wrapper__observation_spec(
    mocker: pytest_mock.MockerFixture,
    wrapped_fake_environment: Wrapper,
    fake_environment: FakeEnvironment,
) -> None:
    """Checks `Wrapper.observation_spec` calls the observation_spec function of the underlying env."""
    mock_obs_spec = mocker.patch.object(
        fake_environment, "observation_spec", autospec=True
    )

    wrapped_fake_environment.observation_spec()

    mock_obs_spec.assert_called_once()


def test_wrapper__action_spec(
    mocker: pytest_mock.MockerFixture,
    wrapped_fake_environment: Wrapper,
    fake_environment: FakeEnvironment,
) -> None:
    """Checks `Wrapper.action_spec` calls the action_spec function of the underlying env."""
    mock_action_spec = mocker.patch.object(
        fake_environment, "action_spec", autospec=True
    )

    wrapped_fake_environment.action_spec()

    mock_action_spec.assert_called_once()


def test_wrapper__repr(wrapped_fake_environment: Wrapper) -> None:
    """Checks `Wrapper.__repr__` returns the expected representation string."""
    repr_str = repr(wrapped_fake_environment)
    assert "MockWrapper" in repr_str


def test_wrapper__render(
    mocker: pytest_mock.MockerFixture,
    wrapped_fake_environment: Wrapper,
    fake_environment: FakeEnvironment,
) -> None:
    """Checks `Wrapper.render` calls the render method of the underlying env."""

    mock_action_spec = mocker.patch.object(fake_environment, "render", autospec=True)

    mock_state = mocker.MagicMock()
    wrapped_fake_environment.render(mock_state)

    mock_action_spec.assert_called_once()


def test_wrapper__close(
    mocker: pytest_mock.MockerFixture,
    wrapped_fake_environment: Wrapper,
    fake_environment: FakeEnvironment,
) -> None:
    """Checks `Wrapper.close` calls the close method of the underlying env."""

    mock_action_spec = mocker.patch.object(fake_environment, "close", autospec=True)

    wrapped_fake_environment.close()

    mock_action_spec.assert_called_once()


def test_wrapper__getattr(
    wrapped_fake_environment: Wrapper, fake_environment: FakeEnvironment
) -> None:
    """Checks `Wrapper.__getattr__` calls the underlying env for unknown attr."""
    # time_limit is defined in the mock env
    assert wrapped_fake_environment.time_limit == fake_environment.time_limit
