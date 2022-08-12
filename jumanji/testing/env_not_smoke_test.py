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

import chex
import jax.numpy as jnp
import jax.random
import pytest

from jumanji.testing.env_not_smoke import (
    SelectActionFn,
    check_env_does_not_smoke,
    make_random_select_action_fn,
)
from jumanji.testing.fakes import FakeEnvironment


@pytest.fixture
def fake_env() -> FakeEnvironment:
    """Instantiate the FakeEnvironment for testing."""
    return FakeEnvironment()


@pytest.fixture
def invalid_select_action_fn() -> SelectActionFn:
    def select_action(key: chex.PRNGKey, state: chex.ArrayTree) -> chex.ArrayTree:
        del state, key
        return jnp.ones(1007)  # an invalid action

    return select_action


def test_env_not_smoke(
    fake_env: FakeEnvironment, invalid_select_action_fn: SelectActionFn
) -> None:
    """Test that the test_env_not_smoke function raises not errors for a valid environment, and
    raises errors for an invalid environment."""
    check_env_does_not_smoke(fake_env)
    with pytest.raises(Exception):
        check_env_does_not_smoke(fake_env, invalid_select_action_fn)


def test_random_select_action(fake_env: FakeEnvironment) -> None:
    """Validate that the `select_action` method returns random actions meeting the
    environment spec."""
    key = jax.random.PRNGKey(0)
    select_action = make_random_select_action_fn(fake_env.action_spec())
    key1, key2, key3 = jax.random.split(key, 3)
    env_state, timestep, extra = fake_env.reset(key1)
    action_1 = select_action(key2, timestep.observation)
    action_2 = select_action(key3, timestep.observation)
    # TODO: uncomment after 64/32 bit issue is fixed
    # fake_env.action_spec().validate(action_1)
    assert not (action_1 == action_2).all()
