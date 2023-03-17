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
from typing import Union

import dm_env
import jax.numpy as jnp
import pytest
from jax import lax

from jumanji.types import (
    StepType,
    TimeStep,
    get_valid_dtype,
    restart,
    termination,
    transition,
    truncation,
)


def test_dm_env_timestep__not_jaxtype() -> None:
    """Validates that dm_env.TimeStep is not a valid Jax type, and hence cannot
    be used in lax.select or lax.cond. Indeed, when not using a Jax type, both lax.select and
    lax.cond fail. This test checks the behavior of neither termination nor transition, but rather
    makes sure that dm_env.TimeStep is not a Jax type. If it is, the latter can be used in
    replacement of the custom implementation of TimeStep.
    """

    def get_termination_transition() -> dm_env.TimeStep:
        """Returns either a termination or transition TimeStep."""
        return lax.select(
            True,
            dm_env.termination(reward=None, observation=None),
            dm_env.transition(reward=None, observation=None),
        )

    try:
        _ = get_termination_transition()
        raise EnvironmentError(
            "dm_env.TimeStep now seems to be a Jax type, meaning it can replace "
            "custom implementation of TimeStep."
        )
    except Exception as ex:
        assert isinstance(ex, TypeError)


@pytest.mark.parametrize("done", (True, False))
def test_timestep__jaxtype(done: bool) -> None:
    """Validates that custom TimeStep from jumanji.types is a valid Jax type.
    Checks that custom termination, transition, restart and truncation functions return
    a valid Jax type. This test checks the behavior of neither termination, transition,
    truncation nor restart, but rather makes sure that TimeStep is a Jax type.
    """

    def get_termination_transition() -> TimeStep:
        """Returns either a termination or transition TimeStep."""
        timestep_termination_transition: TimeStep = lax.cond(
            done,
            lambda _: termination(
                reward=jnp.zeros((), float), observation=jnp.zeros((), float)
            ),
            lambda _: transition(
                reward=jnp.zeros((), float), observation=jnp.zeros((), float)
            ),
            None,
        )
        return timestep_termination_transition

    timestep = get_termination_transition()
    assert isinstance(timestep, TimeStep)
    assert not isinstance(timestep, dm_env.TimeStep)

    def get_restart_truncation() -> TimeStep:
        """Returns either a restart or truncation TimeStep."""
        timestep_restart_truncation: TimeStep = lax.cond(
            done,
            lambda _: restart(observation=jnp.zeros((), float)),
            lambda _: truncation(
                reward=jnp.zeros((), float), observation=jnp.zeros((), float)
            ),
            None,
        )
        return timestep_restart_truncation

    timestep = get_restart_truncation()
    assert isinstance(timestep, TimeStep)
    assert not isinstance(timestep, dm_env.TimeStep)


def test_timestep__restart() -> None:
    """Validates that restart function returns the desired TimeStep."""
    observation = jnp.ones(5, float)
    timestep = restart(observation)
    assert jnp.all(timestep.observation == observation)
    assert timestep.step_type == StepType.FIRST
    assert timestep.reward == 0.0
    assert timestep.discount == 1.0


def test_timestep__transition() -> None:
    """Validates that transition function returns the desired TimeStep."""
    observation = jnp.ones(5, float)
    reward = jnp.array(2.0, float)
    timestep = transition(reward, observation)
    assert jnp.all(timestep.observation == observation)
    assert timestep.step_type == StepType.MID
    assert timestep.reward == reward
    assert timestep.discount == 1.0


def test_timestep__truncation() -> None:
    """Validates that truncation function returns the desired TimeStep."""
    observation = jnp.ones(5, float)
    reward = jnp.array(2.0, float)
    discount = jnp.array(0.8, float)
    timestep = truncation(reward, observation, discount)
    assert jnp.all(timestep.observation == observation)
    assert timestep.step_type == StepType.LAST
    assert timestep.reward == reward
    assert timestep.discount == discount


def test_timestep__termination() -> None:
    """Validates that termination function returns the desired TimeStep."""
    observation = jnp.ones(5, float)
    reward = jnp.array(2.0, float)
    timestep = termination(reward, observation)
    assert jnp.all(timestep.observation == observation)
    assert timestep.step_type == StepType.LAST
    assert timestep.reward == reward
    assert timestep.discount == 0.0


class TestMultiAgent:
    num_agents = 3
    observation = jnp.ones((num_agents, 5), float)
    reward = jnp.arange(1, num_agents + 1, dtype=float)
    discount = 0.8 * jnp.ones(num_agents, float)

    def test_timestep__restart_multi_agent(self) -> None:
        """Validates that restart function returns the desired TimeStep in the multi agent case."""
        timestep = restart(self.observation, shape=(self.num_agents,))
        assert jnp.all(timestep.observation == self.observation)
        assert timestep.step_type == StepType.FIRST
        assert jnp.all(timestep.reward == jnp.zeros((self.num_agents,), float))
        assert jnp.all(timestep.discount == jnp.ones((self.num_agents,), float))

    def test_timestep__transition_multi_agent(self) -> None:
        """Validates that transition function returns the desired TimeStep
        in the multi agent case.
        """

        timestep = transition(self.reward, self.observation, shape=(self.num_agents,))
        assert jnp.all(timestep.observation == self.observation)
        assert timestep.step_type == StepType.MID
        assert jnp.all(timestep.reward == self.reward)
        assert jnp.all(timestep.discount == jnp.ones((self.num_agents,), float))

        timestep = transition(self.reward, self.observation, self.discount)
        assert jnp.all(timestep.discount == self.discount)

    def test_timestep__truncation_multi_agent(self) -> None:
        """Validates that truncation function returns the desired TimeStep
        in the multi agent case.
        """

        timestep = truncation(self.reward, self.observation, self.discount)
        assert jnp.all(timestep.observation == self.observation)
        assert timestep.step_type == StepType.LAST
        assert jnp.all(timestep.reward == self.reward)
        assert jnp.all(timestep.discount == self.discount)

    def test_timestep__termination_multi_agent(self) -> None:
        """Validates that termination function returns the desired TimeStep
        in the multi agent case.
        """

        timestep = termination(self.reward, self.observation)
        assert jnp.all(timestep.observation == self.observation)
        assert timestep.step_type == StepType.LAST
        assert jnp.all(timestep.reward == self.reward)
        assert jnp.all(timestep.discount == jnp.zeros((self.num_agents,), float))


@pytest.mark.parametrize(
    "step_type, is_first, is_mid, is_last",
    [
        (StepType.FIRST, True, False, False),
        (StepType.MID, False, True, False),
        (StepType.LAST, False, False, True),
    ],
)
def test_step_type__helpers(
    step_type: StepType, is_first: bool, is_mid: bool, is_last: bool
) -> None:
    """Test TimeStep and StepType methods first, mid and last."""
    time_step = TimeStep(
        reward=jnp.array(0.0, float),
        discount=jnp.array(1.0, float),
        observation=jnp.array(()),
        step_type=step_type,
    )
    # TimeStep methods
    assert time_step.first() == is_first
    assert time_step.mid() == is_mid
    assert time_step.last() == is_last
    # StepType checks
    assert (time_step.step_type == 0) == is_first
    assert (time_step.step_type == 1) == is_mid
    assert (time_step.step_type == 2) == is_last


@pytest.mark.parametrize(
    "dtype, array_dtype",
    [
        (jnp.dtype(float), jnp.array((), float).dtype),
        (jnp.dtype(int), jnp.array((), int).dtype),
    ],
)
def test_get_valid_dtype(dtype: Union[jnp.dtype, type], array_dtype: jnp.dtype) -> None:
    """Test that get_valid_dtype converts dtype(float64) to dtype(float32) when 64 bit is not
    enabled (by default). Does the same for int types.
    """
    assert array_dtype != dtype
    assert array_dtype == get_valid_dtype(dtype)
