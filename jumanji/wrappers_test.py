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

from collections import namedtuple
from typing import Tuple, Type

import chex
import dm_env.specs
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pytest_mock

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.packing.bin_pack import conftest as bin_pack_conftest
from jumanji.environments.packing.bin_pack.env import BinPack
from jumanji.testing.fakes import FakeEnvironment, FakeMultiEnvironment, FakeState
from jumanji.testing.pytrees import assert_trees_are_different
from jumanji.types import StepType, TimeStep
from jumanji.wrappers import (
    NEXT_OBS_KEY_IN_EXTRAS,
    AutoResetWrapper,
    JumanjiToDMEnvWrapper,
    JumanjiToGymWrapper,
    MultiToSingleWrapper,
    VmapAutoResetWrapper,
    VmapWrapper,
    Wrapper,
    jumanji_to_gym_obs,
)

FakeWrapper = Wrapper[FakeState, specs.BoundedArray, chex.Array]


@pytest.fixture
def mock_wrapper_class() -> Type[FakeWrapper]:
    class MockWrapper(Wrapper[FakeState, specs.BoundedArray, chex.Array]):
        pass

    return MockWrapper


@pytest.fixture
def key() -> chex.PRNGKey:
    return jax.random.PRNGKey(0)


@pytest.fixture
def keys() -> chex.PRNGKey:
    return jax.random.split(jax.random.PRNGKey(0), num=5)


class TestBaseWrapper:
    """Test the base Wrapper used for extending functionality of the Jumanji Environment."""

    @pytest.fixture
    def wrapped_fake_environment(
        self, mock_wrapper_class: Type[FakeWrapper], fake_environment: FakeEnvironment
    ) -> FakeWrapper:
        wrapped_env = mock_wrapper_class(fake_environment)
        return wrapped_env

    def test_wrapper__unwrapped(
        self, wrapped_fake_environment: FakeWrapper, fake_environment: FakeEnvironment
    ) -> None:
        """Checks `Wrapper.unwrapped` returns the unwrapped env."""
        assert wrapped_fake_environment.unwrapped is fake_environment

    def test_wrapper__step(
        self,
        mocker: pytest_mock.MockerFixture,
        wrapped_fake_environment: FakeWrapper,
        fake_environment: FakeEnvironment,
    ) -> None:
        """Checks `Wrapper.step` calls the step method of the underlying env."""
        mock_step = mocker.patch.object(fake_environment, "step", autospec=True)
        mock_state = mocker.MagicMock()
        mock_action = mocker.MagicMock()

        wrapped_fake_environment.step(mock_state, mock_action)

        mock_step.assert_called_once_with(mock_state, mock_action)

    def test_wrapper__reset(
        self,
        mocker: pytest_mock.MockerFixture,
        wrapped_fake_environment: FakeWrapper,
        fake_environment: FakeEnvironment,
    ) -> None:
        """Checks `Wrapper.reset` calls the reset method of the underlying env."""
        mock_reset = mocker.patch.object(fake_environment, "reset", autospec=True)
        mock_key = mocker.MagicMock()

        wrapped_fake_environment.reset(mock_key)

        mock_reset.assert_called_once_with(mock_key)

    def test_wrapper__observation_spec(
        self,
        mocker: pytest_mock.MockerFixture,
        mock_wrapper_class: Type[FakeWrapper],
        fake_environment: FakeEnvironment,
    ) -> None:
        """Checks `Wrapper.__init__` calls the observation_spec function of the underlying env."""
        mock_obs_spec = mocker.patch.object(
            FakeEnvironment, "observation_spec", new_callable=mocker.PropertyMock
        )

        wrapped_fake_environment = mock_wrapper_class(fake_environment)
        mock_obs_spec.assert_called_once()

        wrapped_fake_environment.observation_spec  # noqa: B018
        mock_obs_spec.assert_called_once()

    def test_wrapper__action_spec(
        self,
        mocker: pytest_mock.MockerFixture,
        mock_wrapper_class: Type[FakeWrapper],
        fake_environment: FakeEnvironment,
    ) -> None:
        """Checks `Wrapper.__init__` calls the action_spec function of the underlying env."""
        mock_action_spec = mocker.patch.object(
            FakeEnvironment, "action_spec", new_callable=mocker.PropertyMock
        )

        wrapped_fake_environment = mock_wrapper_class(fake_environment)
        mock_action_spec.assert_called_once()

        wrapped_fake_environment.action_spec  # noqa: B018
        mock_action_spec.assert_called_once()

    def test_wrapper__repr(self, wrapped_fake_environment: FakeWrapper) -> None:
        """Checks `Wrapper.__repr__` returns the expected representation string."""
        repr_str = repr(wrapped_fake_environment)
        assert "MockWrapper" in repr_str

    def test_wrapper__render(
        self,
        mocker: pytest_mock.MockerFixture,
        wrapped_fake_environment: FakeWrapper,
        fake_environment: FakeEnvironment,
    ) -> None:
        """Checks `Wrapper.render` calls the render method of the underlying env."""

        mock_action_spec = mocker.patch.object(fake_environment, "render", autospec=True)

        mock_state = mocker.MagicMock()
        wrapped_fake_environment.render(mock_state)

        mock_action_spec.assert_called_once()

    def test_wrapper__close(
        self,
        mocker: pytest_mock.MockerFixture,
        wrapped_fake_environment: FakeWrapper,
        fake_environment: FakeEnvironment,
    ) -> None:
        """Checks `Wrapper.close` calls the close method of the underlying env."""

        mock_action_spec = mocker.patch.object(fake_environment, "close", autospec=True)

        wrapped_fake_environment.close()

        mock_action_spec.assert_called_once()

    def test_wrapper__getattr(
        self, wrapped_fake_environment: FakeWrapper, fake_environment: FakeEnvironment
    ) -> None:
        """Checks `Wrapper.__getattr__` calls the underlying env for unknown attr."""
        # time_limit is defined in the mock env
        assert wrapped_fake_environment.time_limit == fake_environment.time_limit


FakeJumanjiToDMEnvWrapper = JumanjiToDMEnvWrapper[FakeState, specs.BoundedArray, chex.Array]


class TestJumanjiEnvironmentToDeepMindEnv:
    """Test the JumanjiEnvironmentToDeepMindEnv that transforms an Environment into a
    dm_env.Environment format.
    """

    def test_jumanji_environment_to_deep_mind_env__init(
        self,
        fake_environment: FakeEnvironment,
        key: chex.PRNGKey,
    ) -> None:
        """Validates initialization of the dm_env wrapper."""
        dm_environment = JumanjiToDMEnvWrapper(fake_environment)
        assert isinstance(dm_environment, dm_env.Environment)
        dm_environment_with_key = JumanjiToDMEnvWrapper(
            fake_environment,
            key=key,
        )
        assert isinstance(dm_environment_with_key, dm_env.Environment)

    def test_dm_env__reset(self, fake_dm_env: FakeJumanjiToDMEnvWrapper) -> None:
        """Validates reset function and timestep type of the wrapped environment."""
        timestep = fake_dm_env.reset()
        assert isinstance(timestep, dm_env.TimeStep)
        assert timestep.step_type == dm_env.StepType.FIRST

    def test_jumanji_environment_to_deep_mind_env__step(
        self, fake_dm_env: FakeJumanjiToDMEnvWrapper
    ) -> None:
        """Validates step function of the wrapped environment."""
        timestep = fake_dm_env.reset()
        action = fake_dm_env.action_spec().generate_value()
        next_timestep = fake_dm_env.step(action)
        assert next_timestep != timestep

    def test_jumanji_environment_to_deep_mind_env__observation_spec(
        self, fake_dm_env: FakeJumanjiToDMEnvWrapper
    ) -> None:
        """Validates observation_spec property of the wrapped environment."""
        assert isinstance(fake_dm_env.observation_spec(), dm_env.specs.Array)

    def test_jumanji_environment_to_deep_mind_env__action_spec(
        self, fake_dm_env: FakeJumanjiToDMEnvWrapper
    ) -> None:
        """Validates action_spec property of the wrapped environment."""
        assert isinstance(fake_dm_env.action_spec(), dm_env.specs.Array)

    def test_jumanji_environment_to_deep_mind_env__unwrapped(
        self, fake_dm_env: FakeJumanjiToDMEnvWrapper
    ) -> None:
        """Validates unwrapped property of the wrapped environment."""
        assert isinstance(fake_dm_env.unwrapped, Environment)


FakeJumanjiToGymWrapper = JumanjiToGymWrapper[FakeState, specs.BoundedArray, chex.Array]


class TestJumanjiEnvironmentToGymEnv:
    """
    Test the JumanjiEnvironmentToGymEnv that transforms an Environment into a gym.Env format.
    """

    @pytest.fixture
    def fake_gym_env(self, time_limit: int = 10) -> FakeJumanjiToGymWrapper:
        """Creates a fake environment wrapped as a gym.Env."""
        return JumanjiToGymWrapper(FakeEnvironment(time_limit=time_limit))

    def test_jumanji_environment_to_gym_env__init(self, fake_environment: FakeEnvironment) -> None:
        """Validates initialization of the gym wrapper."""
        gym_environment = JumanjiToGymWrapper(fake_environment)
        assert isinstance(gym_environment, gym.Env)
        gym_environment_with_seed = JumanjiToGymWrapper(fake_environment, seed=0)
        assert isinstance(gym_environment_with_seed, gym.Env)

    def test_jumanji_environment_to_gym_env__reset(
        self, fake_gym_env: FakeJumanjiToGymWrapper
    ) -> None:
        """Validates reset function of the wrapped environment."""
        observation1, info1 = fake_gym_env.reset()
        state1 = fake_gym_env._state
        observation2, info2 = fake_gym_env.reset()
        state2 = fake_gym_env._state

        # Observation is typically numpy array
        assert isinstance(observation1, chex.ArrayNumpy)
        assert isinstance(observation2, chex.ArrayNumpy)

        assert isinstance(info1, dict)
        assert isinstance(info2, dict)

        # Check that the observations are equal
        chex.assert_trees_all_equal(observation1, observation2)
        assert_trees_are_different(state1, state2)

    def test_jumanji_environment_to_gym_env__step(
        self, fake_gym_env: FakeJumanjiToGymWrapper
    ) -> None:
        """Validates step function of the wrapped environment."""
        observation, _ = fake_gym_env.reset()
        action = fake_gym_env.action_space.sample()
        next_observation, reward, terminated, truncated, info = fake_gym_env.step(action)
        assert_trees_are_different(observation, next_observation)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_jumanji_environment_to_gym_env__observation_space(
        self, fake_gym_env: FakeJumanjiToGymWrapper
    ) -> None:
        """Validates observation_space attribute of the wrapped environment."""
        assert isinstance(fake_gym_env.observation_space, gym.spaces.Space)

    def test_jumanji_environment_to_gym_env__action_space(
        self, fake_gym_env: FakeJumanjiToGymWrapper
    ) -> None:
        """Validates action_space attribute of the wrapped environment."""
        assert isinstance(fake_gym_env.action_space, gym.spaces.Space)

    def test_jumanji_environment_to_gym_env__render(
        self,
        mocker: pytest_mock.MockerFixture,
        fake_gym_env: FakeJumanjiToGymWrapper,
    ) -> None:
        mock_render = mocker.patch.object(fake_gym_env.unwrapped, "render", autospec=True)

        mock_state = mocker.MagicMock()
        with pytest.raises(ValueError):
            fake_gym_env.render(mock_state)
        fake_gym_env.reset()
        fake_gym_env.render(mock_state)

        mock_render.assert_called_once()

    def test_jumanji_environment_to_gym_env__close(
        self,
        mocker: pytest_mock.MockerFixture,
        fake_gym_env: FakeJumanjiToGymWrapper,
    ) -> None:
        mock_close = mocker.patch.object(fake_gym_env.unwrapped, "close", autospec=True)

        fake_gym_env.close()

        mock_close.assert_called_once()

    def test_jumanji_environment_to_gym_env__unwrapped(
        self, fake_gym_env: FakeJumanjiToGymWrapper
    ) -> None:
        """Validates unwrapped property of the wrapped environment."""
        assert isinstance(fake_gym_env.unwrapped, Environment)


FakeMultiToSingleWrapper = MultiToSingleWrapper[FakeState, specs.BoundedArray, chex.Array]


class TestMultiToSingleEnvironment:
    @pytest.fixture
    def fake_multi_to_single_env(
        self, fake_multi_environment: FakeMultiEnvironment
    ) -> FakeMultiToSingleWrapper:
        """Creates a fake wrapper that converts a multi-agent Environment
        to a single-agent Environment."""
        return MultiToSingleWrapper(fake_multi_environment)

    def test_multi_env_wrapper__init(
        self,
        fake_multi_environment: FakeMultiEnvironment,
        fake_multi_to_single_env: FakeMultiToSingleWrapper,
    ) -> None:
        """Validates initialization of the multi agent to single agent wrapper."""
        single_agent_env = MultiToSingleWrapper(fake_multi_environment)
        assert isinstance(single_agent_env, Environment)

    def test_multi_env__reset(
        self,
        fake_multi_environment: FakeMultiEnvironment,
        fake_multi_to_single_env: FakeMultiToSingleWrapper,
        key: chex.PRNGKey,
    ) -> None:
        """Validates (jitted) reset function and timestep type of the multi agent
        to single agent wrapped environment.
        """
        _, timestep = jax.jit(fake_multi_to_single_env.reset)(key)
        assert isinstance(timestep, TimeStep)
        assert timestep.step_type == StepType.FIRST
        assert timestep.observation.shape[0] == fake_multi_environment.num_agents
        assert timestep.reward.shape == ()
        assert timestep.discount.shape == ()

    def test_multi_env__step(
        self,
        fake_multi_environment: FakeMultiEnvironment,
        fake_multi_to_single_env: FakeMultiToSingleWrapper,
        key: chex.PRNGKey,
    ) -> None:
        """Validates (jitted) step function of the multi agent to single
        agent wrapped environment.
        """
        state, timestep = fake_multi_to_single_env.reset(key)
        action = fake_multi_to_single_env.action_spec.generate_value()
        state, next_timestep = jax.jit(fake_multi_to_single_env.step)(state, action)
        assert next_timestep != timestep
        assert next_timestep.reward.shape == ()
        assert (
            next_timestep.reward
            == fake_multi_environment.reward_per_step * fake_multi_environment.num_agents
        )
        assert next_timestep.discount.shape == ()
        assert next_timestep.observation.shape[0] == fake_multi_environment.num_agents

    def test_multi_env__different_reward_aggregator(
        self,
        fake_multi_environment: FakeMultiEnvironment,
        fake_multi_to_single_env: FakeMultiToSingleWrapper,
        key: chex.PRNGKey,
    ) -> None:
        """Checks that using a different reward aggregator is correct."""
        mean_fake_multi_to_single_env = MultiToSingleWrapper(
            fake_multi_environment, reward_aggregator=jnp.mean
        )
        state, timestep = mean_fake_multi_to_single_env.reset(key)
        action = mean_fake_multi_to_single_env.action_spec.generate_value()
        state, next_timestep = mean_fake_multi_to_single_env.step(state, action)
        assert next_timestep != timestep
        assert next_timestep.reward.shape == ()
        assert next_timestep.reward == fake_multi_environment.reward_per_step
        assert next_timestep.discount.shape == ()
        assert next_timestep.observation.shape[0] == fake_multi_environment.num_agents

    def test_multi_env__observation_spec(
        self,
        fake_multi_environment: FakeMultiEnvironment,
        fake_multi_to_single_env: FakeMultiToSingleWrapper,
    ) -> None:
        """Validates observation_spec property of the multi agent to single
        agent wrapped environment.
        """
        obs_spec: specs.Array = fake_multi_to_single_env.observation_spec  # type: ignore
        assert isinstance(obs_spec, specs.Array)
        multi_obs_spec: specs.Array = fake_multi_environment.observation_spec  # type: ignore
        assert obs_spec.shape == multi_obs_spec.shape

    def test_multi_env__action_spec(
        self,
        fake_multi_environment: FakeMultiEnvironment,
        fake_multi_to_single_env: FakeMultiToSingleWrapper,
    ) -> None:
        """Validates action_spec property of the multi agent to single
        agent wrapped environment.
        """
        action_spec = fake_multi_to_single_env.action_spec
        assert isinstance(fake_multi_to_single_env.action_spec, specs.Array)
        assert action_spec.shape == fake_multi_environment.action_spec.shape

    def test_multi_env__unwrapped(
        self,
        fake_multi_environment: FakeMultiEnvironment,
        fake_multi_to_single_env: FakeMultiToSingleWrapper,
    ) -> None:
        """Validates unwrapped property of the multi agent to single
        agent wrapped environment.
        """
        assert isinstance(fake_multi_to_single_env.unwrapped, Environment)
        assert fake_multi_to_single_env._env is fake_multi_environment


FakeVmapWrapper = Wrapper[FakeState, specs.BoundedArray, chex.Array]


class TestVmapWrapper:
    @pytest.fixture
    def fake_vmap_environment(self, fake_environment: FakeEnvironment) -> FakeVmapWrapper:
        return VmapWrapper(fake_environment)

    def test_vmap_wrapper__init(self, fake_environment: FakeEnvironment) -> None:
        """Validates initialization of the vmap wrapper."""
        vmap_env = VmapWrapper(fake_environment)
        assert isinstance(vmap_env, Environment)

    def test_vmap_env__reset(
        self, fake_vmap_environment: FakeVmapWrapper, keys: chex.PRNGKey
    ) -> None:
        """Validates reset function and timestep type of the vmap wrapped environment."""
        _, timestep = jax.jit(fake_vmap_environment.reset)(keys)

        assert isinstance(timestep, TimeStep)
        chex.assert_trees_all_equal(timestep.step_type, StepType.FIRST)
        assert timestep.observation.shape[0] == keys.shape[0]
        assert timestep.reward.shape == (keys.shape[0],)
        assert timestep.discount.shape == (keys.shape[0],)

    def test_vmap_env__step(
        self, fake_vmap_environment: FakeVmapWrapper, keys: chex.PRNGKey
    ) -> None:
        """Validates step function of the vmap environment."""
        state, timestep = fake_vmap_environment.reset(keys)
        action = jax.vmap(lambda _: fake_vmap_environment.action_spec.generate_value())(keys)

        state, next_timestep = jax.jit(fake_vmap_environment.step)(state, action)

        assert_trees_are_different(next_timestep, timestep)
        chex.assert_trees_all_equal(next_timestep.reward, 0)

        assert next_timestep.reward.shape == (keys.shape[0],)
        assert next_timestep.discount.shape == (keys.shape[0],)
        assert next_timestep.observation.shape[0] == keys.shape[0]

    def test_vmap_env__render(
        self, fake_vmap_environment: FakeVmapWrapper, keys: chex.PRNGKey
    ) -> None:
        states, _ = fake_vmap_environment.reset(keys)
        result = fake_vmap_environment.render(states)
        assert result == (keys.shape[1:], ())

    def test_vmap_env__unwrapped(
        self, fake_environment: FakeEnvironment, fake_vmap_environment: FakeVmapWrapper
    ) -> None:
        """Validates unwrapped property of the vmap environment."""
        assert isinstance(fake_vmap_environment.unwrapped, Environment)
        assert fake_vmap_environment._env is fake_environment


FakeAutoResetWrapper = AutoResetWrapper[FakeState, specs.BoundedArray, chex.Array]


class TestAutoResetWrapper:
    @pytest.fixture
    def fake_auto_reset_environment(
        self, fake_environment: FakeEnvironment
    ) -> FakeAutoResetWrapper:
        return AutoResetWrapper(fake_environment, next_obs_in_extras=True)

    @pytest.fixture
    def fake_state_and_timestep(
        self, fake_auto_reset_environment: FakeAutoResetWrapper, key: chex.PRNGKey
    ) -> Tuple[FakeState, TimeStep[chex.Array]]:
        state, timestep = jax.jit(fake_auto_reset_environment.reset)(key)
        return state, timestep

    def test_auto_reset_wrapper__init(self, fake_environment: FakeEnvironment) -> None:
        """Validates initialization of the AutoResetWrapper."""
        auto_reset_env = AutoResetWrapper(fake_environment)
        assert isinstance(auto_reset_env, Environment)

    def test_auto_reset_wrapper__auto_reset(
        self,
        fake_auto_reset_environment: FakeAutoResetWrapper,
        fake_state_and_timestep: Tuple[FakeState, TimeStep[chex.Array]],
    ) -> None:
        """Validates the auto_reset function of the AutoResetWrapper."""
        state, timestep = fake_state_and_timestep
        _, reset_timestep = jax.jit(fake_auto_reset_environment._auto_reset)(state, timestep)
        chex.assert_trees_all_equal(timestep.observation, reset_timestep.observation)
        # Expect that non-reset timestep obs and extras are the same.
        assert jnp.all(timestep.observation == timestep.extras[NEXT_OBS_KEY_IN_EXTRAS])

    def test_auto_reset_wrapper__step_no_reset(
        self, fake_auto_reset_environment: FakeAutoResetWrapper, key: chex.PRNGKey
    ) -> None:
        """Validates that step function of the AutoResetWrapper does not do an
        auto-reset when the terminal state is not reached.
        """
        state, first_timestep = fake_auto_reset_environment.reset(key)

        # Generate an action
        action = fake_auto_reset_environment.action_spec.generate_value()

        state, timestep = jax.jit(fake_auto_reset_environment.step)(state, action)  # type: Tuple[FakeState, TimeStep[chex.Array]]

        assert timestep.step_type == StepType.MID
        assert_trees_are_different(timestep, first_timestep)
        chex.assert_trees_all_equal(timestep.reward, 0)
        # no reset so expect extras and obs to be the same.
        assert jnp.all(timestep.observation == timestep.extras[NEXT_OBS_KEY_IN_EXTRAS])

    def test_auto_reset_wrapper__step_reset(
        self,
        fake_environment: FakeEnvironment,
        fake_auto_reset_environment: FakeAutoResetWrapper,
        key: chex.PRNGKey,
    ) -> None:
        """Validates that the auto-reset is done correctly by the step function
        of the AutoResetWrapper when the terminal timestep is reached.
        """
        state, first_timestep = fake_auto_reset_environment.reset(key)  # type: ignore

        fake_environment.time_limit = 5

        # Loop across time_limit so auto-reset occurs
        for _ in range(fake_environment.time_limit - 1):
            action = fake_auto_reset_environment.action_spec.generate_value()
            state, timestep = jax.jit(fake_auto_reset_environment.step)(state, action)
            assert jnp.all(timestep.observation == timestep.extras[NEXT_OBS_KEY_IN_EXTRAS])

        state, final_timestep = jax.jit(fake_auto_reset_environment.step)(state, action)

        assert final_timestep.step_type == StepType.LAST
        chex.assert_trees_all_equal(final_timestep.observation, first_timestep.observation)
        assert not jnp.all(
            final_timestep.observation == final_timestep.extras[NEXT_OBS_KEY_IN_EXTRAS]
        )
        assert jnp.all((timestep.observation + 1) == final_timestep.extras[NEXT_OBS_KEY_IN_EXTRAS])


FakeVmapAutoResetWrapper = VmapAutoResetWrapper[FakeState, specs.BoundedArray, chex.Array]


class TestVmapAutoResetWrapper:
    @pytest.fixture
    def fake_vmap_auto_reset_environment(
        self, fake_environment: FakeEnvironment
    ) -> FakeVmapAutoResetWrapper:
        return VmapAutoResetWrapper(fake_environment, next_obs_in_extras=True)

    @pytest.fixture
    def action(
        self,
        fake_vmap_auto_reset_environment: FakeVmapAutoResetWrapper,
        keys: chex.PRNGKey,
    ) -> chex.Array:
        generate_action_fn = lambda _: fake_vmap_auto_reset_environment.action_spec.generate_value()
        return jax.vmap(generate_action_fn)(keys)

    def test_vmap_auto_reset_wrapper__init(self, fake_environment: FakeEnvironment) -> None:
        """Validates initialization of the wrapper."""
        vmap_auto_reset_env = VmapWrapper(fake_environment)
        assert isinstance(vmap_auto_reset_env, Environment)

    def test_vmap_auto_reset_wrapper__reset(
        self,
        fake_vmap_auto_reset_environment: FakeVmapAutoResetWrapper,
        keys: chex.PRNGKey,
    ) -> None:
        """Validates reset function and timestep type of the wrapper."""
        _, timestep = jax.jit(fake_vmap_auto_reset_environment.reset)(keys)

        assert isinstance(timestep, TimeStep)
        chex.assert_trees_all_equal(timestep.step_type, StepType.FIRST)
        assert timestep.observation.shape[0] == keys.shape[0]
        assert timestep.reward.shape == (keys.shape[0],)
        assert timestep.discount.shape == (keys.shape[0],)
        # only reset so expect extras and obs to be the same.
        assert jnp.all(timestep.observation == timestep.extras[NEXT_OBS_KEY_IN_EXTRAS])

    def test_vmap_auto_reset_wrapper__auto_reset(
        self,
        fake_vmap_auto_reset_environment: FakeVmapAutoResetWrapper,
        keys: chex.PRNGKey,
    ) -> None:
        """Validates the auto_reset function of the wrapper."""
        state, timestep = fake_vmap_auto_reset_environment.reset(keys)
        _, reset_timestep = jax.lax.map(
            lambda args: fake_vmap_auto_reset_environment._auto_reset(*args),
            (state, timestep),
        )
        chex.assert_trees_all_equal(timestep.observation, reset_timestep.observation)
        # expect rest timestep.extras to have the same obs as the original timestep
        assert jnp.all(timestep.observation == reset_timestep.extras[NEXT_OBS_KEY_IN_EXTRAS])

    def test_vmap_auto_reset_wrapper__maybe_reset(
        self,
        fake_vmap_auto_reset_environment: FakeVmapAutoResetWrapper,
        keys: chex.PRNGKey,
    ) -> None:
        """Validates the auto_reset function of the wrapper."""
        state, timestep = fake_vmap_auto_reset_environment.reset(keys)
        _, reset_timestep = jax.lax.map(
            lambda args: fake_vmap_auto_reset_environment._maybe_reset(*args),
            (state, timestep),
        )
        chex.assert_trees_all_equal(timestep.observation, reset_timestep.observation)
        # expect rest timestep.extras to have the same obs as the original timestep
        assert jnp.all(timestep.observation == reset_timestep.extras[NEXT_OBS_KEY_IN_EXTRAS])

    def test_vmap_auto_reset_wrapper__step_no_reset(
        self,
        fake_vmap_auto_reset_environment: FakeVmapAutoResetWrapper,
        keys: chex.PRNGKey,
        action: chex.Array,
    ) -> None:
        """Validates that step function of the wrapper does not do an
        auto-reset when the terminal state is not reached.
        """
        state, first_timestep = fake_vmap_auto_reset_environment.reset(keys)
        state, timestep = jax.jit(fake_vmap_auto_reset_environment.step)(state, action)

        assert jnp.all(timestep.step_type == StepType.MID)
        assert_trees_are_different(timestep, first_timestep)
        chex.assert_trees_all_equal(timestep.reward, 0)

        # no reset so expect extras and obs to be the same.
        # and the first timestep should have different obs in extras.
        assert not jnp.all(first_timestep.observation == timestep.extras[NEXT_OBS_KEY_IN_EXTRAS])
        assert jnp.all(timestep.observation == timestep.extras[NEXT_OBS_KEY_IN_EXTRAS])

    def test_vmap_auto_reset_wrapper__step_reset(
        self,
        fake_environment: FakeEnvironment,
        fake_vmap_auto_reset_environment: FakeVmapAutoResetWrapper,
        keys: chex.PRNGKey,
        action: chex.Array,
    ) -> None:
        """Validates that the auto-reset is done correctly by the step function
        of the wrapper when the terminal timestep is reached.
        """
        state, first_timestep = fake_vmap_auto_reset_environment.reset(keys)
        fake_vmap_auto_reset_environment.unwrapped.time_limit = 5  # type: ignore

        # Loop across time_limit so auto-reset occurs
        for _ in range(fake_vmap_auto_reset_environment.time_limit - 1):
            state, timestep = jax.jit(fake_vmap_auto_reset_environment.step)(state, action)
            assert jnp.all(timestep.observation == timestep.extras[NEXT_OBS_KEY_IN_EXTRAS])

        state, final_timestep = jax.jit(fake_vmap_auto_reset_environment.step)(state, action)
        assert jnp.all(final_timestep.step_type == StepType.LAST)
        chex.assert_trees_all_equal(final_timestep.observation, first_timestep.observation)
        assert not jnp.all(
            final_timestep.observation == final_timestep.extras[NEXT_OBS_KEY_IN_EXTRAS]
        )
        assert jnp.all((timestep.observation + 1) == final_timestep.extras[NEXT_OBS_KEY_IN_EXTRAS])

    def test_vmap_auto_reset_wrapper__step(
        self,
        fake_vmap_auto_reset_environment: FakeVmapAutoResetWrapper,
        keys: chex.PRNGKey,
        action: chex.Array,
    ) -> None:
        """Validates step function of the vmap environment."""
        state, timestep = fake_vmap_auto_reset_environment.reset(keys)
        state, next_timestep = jax.jit(fake_vmap_auto_reset_environment.step)(state, action)

        assert_trees_are_different(next_timestep, timestep)
        chex.assert_trees_all_equal(next_timestep.reward, 0)
        assert next_timestep.reward.shape == (keys.shape[0],)
        assert next_timestep.discount.shape == (keys.shape[0],)
        assert next_timestep.observation.shape[0] == keys.shape[0]
        # expect observation and extras to be the same, since no reset
        assert jnp.all(next_timestep.observation == next_timestep.extras[NEXT_OBS_KEY_IN_EXTRAS])

    def test_vmap_auto_reset_wrapper__render(
        self,
        fake_vmap_auto_reset_environment: FakeVmapAutoResetWrapper,
        keys: chex.PRNGKey,
    ) -> None:
        states, _ = fake_vmap_auto_reset_environment.reset(keys)
        result = fake_vmap_auto_reset_environment.render(states)
        assert result == (keys.shape[1:], ())

    def test_vmap_auto_reset_wrapper__unwrapped(
        self,
        fake_environment: FakeEnvironment,
        fake_vmap_auto_reset_environment: FakeVmapAutoResetWrapper,
    ) -> None:
        """Validates unwrapped property of the vmap environment."""
        assert isinstance(fake_vmap_auto_reset_environment.unwrapped, FakeEnvironment)
        assert fake_vmap_auto_reset_environment._env is fake_environment


class TestJumanjiToGymObservation:
    """Tests for checking the behaviour of jumanji_to_gym_obs."""

    @chex.dataclass
    class DummyChexDataclass:
        x: chex.Array
        y: chex.Array

    def test_jumanji_to_gym_obs__correct_observation(self) -> None:
        """Check that a NamedTuple containing a JAX array and a chex dataclass of two
        JAX arrays is converted correctly into a nested dictionary of numpy arrays.
        """
        NestedObservation = namedtuple("NestedObservation", ["jax_array", "chex_dataclass"])
        array = jnp.zeros((2, 2))
        data_class = self.DummyChexDataclass(x=array, y=array)  # type: ignore
        nested_obs = NestedObservation(array, data_class)

        converted_obs = jumanji_to_gym_obs(nested_obs)
        correct_obs = {
            "jax_array": np.zeros((2, 2)),
            "chex_dataclass": {"x": np.zeros((2, 2)), "y": np.zeros((2, 2))},
        }

        chex.assert_trees_all_equal(converted_obs, correct_obs)
        assert isinstance(converted_obs, dict)
        assert isinstance(converted_obs["jax_array"], chex.ArrayNumpy)
        assert isinstance(converted_obs["chex_dataclass"], dict)
        assert isinstance(converted_obs["chex_dataclass"]["x"], chex.ArrayNumpy)
        assert isinstance(converted_obs["chex_dataclass"]["y"], chex.ArrayNumpy)

    def test_jumanji_to_gym_obs__wrong_observation(self) -> None:
        """Check that a NotImplementedError is raised when the wrong datatype is passed
        to one of the two attributes of the chex dataclass.
        """
        NestedObservation = namedtuple("NestedObservation", ["jax_array", "chex_dataclass"])
        array = jnp.zeros((10, 10))

        # Pass in the wrong datatype
        data_class = self.DummyChexDataclass(x=array, y="array")  # type: ignore
        nested_obs = NestedObservation(array, data_class)

        # Check that the function raises a NotImplementedError
        with pytest.raises(NotImplementedError):
            jumanji_to_gym_obs(nested_obs)

    def test_jumanji_to_gym_obs__bin_pack(self) -> None:
        """Check that an example bin_pack observation is correctly converted."""
        env = BinPack(generator=bin_pack_conftest.DummyGenerator(), obs_num_ems=1)
        obs = env.observation_spec.generate_value()

        converted_obs = jumanji_to_gym_obs(obs)
        correct_obs = {
            "ems": {
                "x1": jnp.zeros((1,), jnp.float32),
                "x2": jnp.zeros((1,), jnp.float32),
                "y1": jnp.zeros((1,), jnp.float32),
                "y2": jnp.zeros((1,), jnp.float32),
                "z1": jnp.zeros((1,), jnp.float32),
                "z2": jnp.zeros((1,), jnp.float32),
            },
            "ems_mask": jnp.bool_([0]),
            "items": {
                "x_len": jnp.zeros((3,), jnp.float32),
                "y_len": jnp.zeros((3,), jnp.float32),
                "z_len": jnp.zeros((3,), jnp.float32),
            },
            "items_mask": jnp.bool_([0, 0, 0]),
            "items_placed": jnp.bool_([0, 0, 0]),
            "action_mask": jnp.bool_([[0, 0, 0]]),
        }
        chex.assert_trees_all_equal(converted_obs, correct_obs)
        assert isinstance(converted_obs, dict)
