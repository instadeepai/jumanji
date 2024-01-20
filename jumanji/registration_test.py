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

from typing import Tuple

import pytest
import pytest_mock

import jumanji
from jumanji import registration, specs
from jumanji.testing.fakes import FakeEnvironment


class TestParser:
    @pytest.mark.parametrize("env_id", ("Env", "Env_v0"))
    def test_parser__wrong_version(self, env_id: str) -> None:
        with pytest.raises(ValueError):
            registration.parse_env_id(env_id)

    @pytest.mark.parametrize(
        "env_id, expected",
        [("Env-v0", ("Env", 0)), ("Env-test-v10", ("Env-test", 10))],
    )
    def test_parser__name_version(self, env_id: str, expected: Tuple[str, int]) -> None:
        assert registration.parse_env_id(env_id) == expected


class TestRegistrationRules:
    def test_registration__next_version(
        self, mocker: pytest_mock.MockerFixture
    ) -> None:
        mocker.patch("jumanji.registration._REGISTRY", {})

        # Check that the next registrable version is v+1
        registration.register("Env-v0", entry_point="")

        env_spec = registration.EnvSpec(id="Env-v2", entry_point="")
        registration._check_registration_is_allowed(env_spec)

        env_spec = registration.EnvSpec(id="Env-v1", entry_point="")
        registration._check_registration_is_allowed(env_spec)

    def test_registration__already_registered(
        self, mocker: pytest_mock.MockerFixture
    ) -> None:
        mocker.patch("jumanji.registration._REGISTRY", {})
        env_spec = registration.EnvSpec(id="Env-v0", entry_point="")
        registration.register(env_spec.id, entry_point=env_spec.entry_point)
        with pytest.raises(ValueError, match="override the registered environment"):
            registration._check_registration_is_allowed(env_spec)


def test_register(mocker: pytest_mock.MockerFixture) -> None:
    mocker.patch("jumanji.registration._REGISTRY", {})
    env_ids = ("Cat-v0", "Dog-v0", "Fish-v0", "Cat-v1")
    for env_id in env_ids:
        registration.register(env_id, entry_point="")
    registered_envs = registration.registered_environments()
    assert all(env_id in registered_envs for env_id in env_ids)


def test_register__instantiate_registered_env(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch("jumanji.registration._REGISTRY", {})
    env_id = "Fake-v0"
    registration.register(
        id=env_id,
        entry_point="jumanji.testing.fakes:FakeEnvironment",
    )
    env = registration.make(env_id)
    assert isinstance(env, FakeEnvironment)


def test_register__override_kwargs(mocker: pytest_mock.MockerFixture) -> None:
    mocker.patch("jumanji.registration._REGISTRY", {})
    env_id = "Fake-v0"
    obs_shape = (11, 17)
    registration.register(
        id=env_id,
        entry_point="jumanji.testing.fakes:FakeEnvironment",
    )
    obs_spec: specs.Array = registration.make(  # type: ignore
        env_id, observation_shape=obs_shape
    ).observation_spec
    assert obs_spec.shape == obs_shape


def test_registration__make() -> None:
    """Check that all the environments in the registry can be initiated correctly
    using `jumanji.make`.
    """
    for env_id, env_spec in registration._REGISTRY.items():
        assert isinstance(env_spec, registration.EnvSpec)

        env = jumanji.make(env_id)
        assert isinstance(env, jumanji.Environment)

        env_class = registration.load(env_spec.entry_point)
        assert isinstance(env, env_class)
