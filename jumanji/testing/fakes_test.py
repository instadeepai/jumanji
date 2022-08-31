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

import pytest
from jax import random

import jumanji.testing.fakes as fakes
from jumanji.types import TimeStep


@pytest.mark.parametrize("fake_environment", [()], indirect=True)
def test_fake_environment__reset(fake_environment: fakes.FakeEnvironment) -> None:
    """Validates the reset of the fake environment."""
    state, timestep = fake_environment.reset(random.PRNGKey(0))
    assert isinstance(state, fakes.FakeState)
    assert isinstance(timestep, TimeStep)


@pytest.mark.parametrize("fake_environment", [()], indirect=True)
def test_fake_environment__step(fake_environment: fakes.FakeEnvironment) -> None:
    """Validates the step function of the fake environment."""
    state, timestep = fake_environment.reset(random.PRNGKey(0))
    action = fake_environment.action_spec().generate_value()
    next_state, timestep = fake_environment.step(state, action)
    # Check that the step value is now different
    assert state.step != next_state.step


@pytest.mark.parametrize("fake_environment", [()], indirect=True)
def test_fake_environment__does_not_smoke(
    fake_environment: fakes.FakeEnvironment,
) -> None:
    """Validates the run of an episode in the fake environment. Check that it does not smoke."""
    state, timestep = fake_environment.reset(random.PRNGKey(0))
    action = fake_environment.action_spec().generate_value()
    while not timestep.last():
        state, timestep = fake_environment.step(state, action)


@pytest.mark.parametrize("fake_multi_environment", [()], indirect=True)
def test_fake_multi_environment__reset(
    fake_multi_environment: fakes.FakeMultiEnvironment,
) -> None:
    """Validates the reset of the fake multi agent environment."""
    state, timestep = fake_multi_environment.reset(random.PRNGKey(0))
    assert isinstance(state, fakes.FakeState)
    assert isinstance(timestep, TimeStep)
    assert timestep.reward.shape == (fake_multi_environment.num_agents,)
    assert timestep.discount.shape == (fake_multi_environment.num_agents,)
    assert timestep.observation.shape[0] == fake_multi_environment.num_agents


@pytest.mark.parametrize("fake_multi_environment", [()], indirect=True)
def test_fake_multi_environment__step(
    fake_multi_environment: fakes.FakeMultiEnvironment,
) -> None:
    """Validates the step function of the fake multi agent environment."""
    state, timestep = fake_multi_environment.reset(random.PRNGKey(0))
    action = fake_multi_environment.action_spec().generate_value()
    assert action.shape[0] == fake_multi_environment.num_agents

    next_state, timestep = fake_multi_environment.step(state, action)
    # Check that the step value is now different
    assert state.step != next_state.step
    assert timestep.reward.shape == (fake_multi_environment.num_agents,)
    assert timestep.discount.shape == (fake_multi_environment.num_agents,)
    assert timestep.observation.shape[0] == fake_multi_environment.num_agents


@pytest.mark.parametrize("fake_multi_environment", [()], indirect=True)
def test_fake_multi_environment__does_not_smoke(
    fake_multi_environment: fakes.FakeMultiEnvironment,
) -> None:
    """Validates the run of an episode in the fake multi agent environment.
    Check that it does not smoke."""
    state, timestep = fake_multi_environment.reset(random.PRNGKey(0))
    action = fake_multi_environment.action_spec().generate_value()
    assert action.shape[0] == fake_multi_environment.num_agents
    while not timestep.last():
        state, timestep = fake_multi_environment.step(state, action)
