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

from typing import Tuple, TypeVar

import brax
import dm_env.specs
import gym
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest
import pytest_mock
from brax.envs import Env as BraxEnv
from brax.envs import State as BraxState
from chex import assert_trees_all_equal

from jumanji import specs
from jumanji.env import Environment
from jumanji.testing.fakes import FakeEnvironment, FakeMultiEnvironment, FakeState
from jumanji.testing.pytrees import assert_trees_are_different
from jumanji.types import Extra, StepType, TimeStep
from jumanji.wrappers import (
    AutoResetWrapper,
    BraxToJumanjiWrapper,
    JumanjiToDMEnvWrapper,
    JumanjiToGymWrapper,
    MultiToSingleWrapper,
    VmapWrapper,
)

State = TypeVar("State")
Observation = TypeVar("Observation")


class TestJumanjiEnvironmentToDeepMindEnv:
    """Test the JumanjiEnvironmentToDeepMindEnv that transforms an Environment into a
    dm_env.Environment format.
    """

    def test_jumanji_environment_to_deep_mind_env__init(
        self, fake_environment: FakeEnvironment
    ) -> None:
        """Validates initialization of the dm_env wrapper."""
        dm_environment = JumanjiToDMEnvWrapper(fake_environment)
        assert isinstance(dm_environment, dm_env.Environment)
        dm_environment_with_key = JumanjiToDMEnvWrapper(
            fake_environment, key=jax.random.PRNGKey(0)
        )
        assert isinstance(dm_environment_with_key, dm_env.Environment)

    def test_dm_env__reset(self, fake_dm_env: JumanjiToDMEnvWrapper) -> None:
        """Validates reset function and timestep type of the wrapped environment."""
        timestep = fake_dm_env.reset()
        assert isinstance(timestep, dm_env.TimeStep)
        assert timestep.step_type == dm_env.StepType.FIRST

    def test_jumanji_environment_to_deep_mind_env__step(
        self, fake_dm_env: JumanjiToDMEnvWrapper
    ) -> None:
        """Validates step function of the wrapped environment."""
        timestep = fake_dm_env.reset()
        action = fake_dm_env.action_spec().generate_value()
        next_timestep = fake_dm_env.step(action)
        assert next_timestep != timestep

    def test_jumanji_environment_to_deep_mind_env__observation_spec(
        self, fake_dm_env: JumanjiToDMEnvWrapper
    ) -> None:
        """Validates observation_spec property of the wrapped environment."""
        assert isinstance(fake_dm_env.observation_spec(), dm_env.specs.Array)

    def test_jumanji_environment_to_deep_mind_env__action_spec(
        self, fake_dm_env: JumanjiToDMEnvWrapper
    ) -> None:
        """Validates action_spec property of the wrapped environment."""
        assert isinstance(fake_dm_env.action_spec(), dm_env.specs.Array)

    def test_jumanji_environment_to_deep_mind_env__unwrapped(
        self, fake_dm_env: JumanjiToDMEnvWrapper
    ) -> None:
        """Validates unwrapped property of the wrapped environment."""
        assert isinstance(fake_dm_env.unwrapped, Environment)


class TestJumanjiEnvironmentToGymEnv:
    """
    Test the JumanjiEnvironmentToGymEnv that transforms an Environment into a gym.Env format.
    """

    @pytest.fixture
    def fake_gym_env(self, time_limit: int = 10) -> gym.Env:
        """Creates a fake environment wrapped as a gym.Env."""
        return JumanjiToGymWrapper(FakeEnvironment(time_limit=time_limit))

    def test_jumanji_environment_to_gym_env__init(
        self, fake_environment: FakeEnvironment
    ) -> None:
        """Validates initialization of the gym wrapper."""
        gym_environment = JumanjiToGymWrapper(fake_environment)
        assert isinstance(gym_environment, gym.Env)
        gym_environment_with_seed = JumanjiToGymWrapper(fake_environment, seed=0)
        assert isinstance(gym_environment_with_seed, gym.Env)

    def test_jumanji_environment_to_gym_env__reset(
        self, fake_gym_env: JumanjiToGymWrapper
    ) -> None:
        """Validates reset function of the wrapped environment."""
        observation1 = fake_gym_env.reset()  # type: ignore
        state1 = fake_gym_env._state
        observation2 = fake_gym_env.reset()  # type: ignore
        state2 = fake_gym_env._state

        # Observation is typically numpy array
        assert isinstance(observation1, np.ndarray)
        assert isinstance(observation2, np.ndarray)

        # Check that the observations are equal
        assert_trees_all_equal(observation1, observation2)
        assert_trees_are_different(state1, state2)

    def test_jumanji_environment_to_gym_env__step(
        self, fake_gym_env: JumanjiToGymWrapper
    ) -> None:
        """Validates step function of the wrapped environment."""
        observation = fake_gym_env.reset()  # type: ignore
        action = fake_gym_env.action_space.sample()
        next_observation, reward, terminated, info = fake_gym_env.step(action)  # type: ignore
        assert_trees_are_different(observation, next_observation)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)

    def test_jumanji_environment_to_gym_env__observation_space(
        self, fake_gym_env: JumanjiToGymWrapper
    ) -> None:
        """Validates observation_space attribute of the wrapped environment."""
        assert isinstance(fake_gym_env.observation_space, gym.spaces.Space)

    def test_jumanji_environment_to_gym_env__action_space(
        self, fake_gym_env: JumanjiToGymWrapper
    ) -> None:
        """Validates action_space attribute of the wrapped environment."""
        assert isinstance(fake_gym_env.action_space, gym.spaces.Space)

    def test_jumanji_environment_to_gym_env__render(
        self,
        mocker: pytest_mock.MockerFixture,
        fake_gym_env: JumanjiToGymWrapper,
    ) -> None:

        mock_render = mocker.patch.object(
            fake_gym_env.unwrapped, "render", autospec=True
        )

        mock_state = mocker.MagicMock()
        fake_gym_env.render(mock_state)

        mock_render.assert_called_once()

    def test_jumanji_environment_to_gym_env__close(
        self,
        mocker: pytest_mock.MockerFixture,
        fake_gym_env: JumanjiToGymWrapper,
    ) -> None:

        mock_close = mocker.patch.object(fake_gym_env.unwrapped, "close", autospec=True)

        fake_gym_env.close()

        mock_close.assert_called_once()

    def test_jumanji_environment_to_gym_env__unwrapped(
        self, fake_gym_env: JumanjiToGymWrapper
    ) -> None:
        """Validates unwrapped property of the wrapped environment."""
        assert isinstance(fake_gym_env.unwrapped, Environment)


class TestMultiToSingleEnvironment:
    @pytest.fixture
    def fake_multi_to_single_env(
        self, fake_multi_environment: FakeMultiEnvironment
    ) -> MultiToSingleWrapper:
        """Creates a fake wrapper that converts a multi-agent Environment
        to a single-agent Environment."""
        return MultiToSingleWrapper(fake_multi_environment)

    def test_multi_env_wrapper__init(
        self,
        fake_multi_environment: FakeMultiEnvironment,
        fake_multi_to_single_env: MultiToSingleWrapper,
    ) -> None:
        """Validates initialization of the multi agent to single agent wrapper."""
        single_agent_env = MultiToSingleWrapper(fake_multi_environment)
        assert isinstance(single_agent_env, Environment)

    def test_multi_env__reset(
        self,
        fake_multi_environment: FakeMultiEnvironment,
        fake_multi_to_single_env: MultiToSingleWrapper,
    ) -> None:
        """Validates (jitted) reset function and timestep type of the multi agent
        to single agent wrapped environment.
        """
        _, timestep, _ = jax.jit(fake_multi_to_single_env.reset)(random.PRNGKey(0))
        assert isinstance(timestep, TimeStep)
        assert timestep.step_type == StepType.FIRST
        assert timestep.observation.shape[0] == fake_multi_environment.num_agents
        assert timestep.reward.shape == ()
        assert timestep.discount.shape == ()

    def test_multi_env__step(
        self,
        fake_multi_environment: FakeMultiEnvironment,
        fake_multi_to_single_env: MultiToSingleWrapper,
    ) -> None:
        """Validates (jitted) step function of the multi agent to single
        agent wrapped environment.
        """
        state, timestep, _ = fake_multi_to_single_env.reset(
            random.PRNGKey(0)
        )  # type: Tuple[FakeState, TimeStep, Extra]
        action = fake_multi_to_single_env.action_spec().generate_value()
        state, next_timestep, _ = jax.jit(fake_multi_to_single_env.step)(
            state, action
        )  # type: Tuple[FakeState, TimeStep, Extra]
        assert next_timestep != timestep
        assert next_timestep.reward.shape == ()
        assert (
            next_timestep.reward
            == fake_multi_environment.reward_per_step
            * fake_multi_environment.num_agents
        )
        assert next_timestep.discount.shape == ()
        assert next_timestep.observation.shape[0] == fake_multi_environment.num_agents

    def test_multi_env__different_reward_aggregator(
        self,
        fake_multi_environment: FakeMultiEnvironment,
        fake_multi_to_single_env: MultiToSingleWrapper,
    ) -> None:
        """Checks that using a different reward aggregator is correct."""
        mean_fake_multi_to_single_env = MultiToSingleWrapper(
            fake_multi_environment, reward_aggregator=jnp.mean
        )
        state, timestep, _ = mean_fake_multi_to_single_env.reset(
            random.PRNGKey(0)
        )  # type: Tuple[FakeState, TimeStep, Extra]
        action = mean_fake_multi_to_single_env.action_spec().generate_value()
        state, next_timestep, _ = mean_fake_multi_to_single_env.step(
            state, action
        )  # type: Tuple[FakeState, TimeStep, Extra]
        assert next_timestep != timestep
        assert next_timestep.reward.shape == ()
        assert next_timestep.reward == fake_multi_environment.reward_per_step
        assert next_timestep.discount.shape == ()
        assert next_timestep.observation.shape[0] == fake_multi_environment.num_agents

    def test_multi_env__observation_spec(
        self,
        fake_multi_environment: FakeMultiEnvironment,
        fake_multi_to_single_env: MultiToSingleWrapper,
    ) -> None:
        """Validates observation_spec property of the multi agent to single
        agent wrapped environment.
        """
        obs_spec: specs.Array = fake_multi_to_single_env.observation_spec()  # type: ignore
        assert isinstance(obs_spec, specs.Array)
        assert obs_spec.shape == fake_multi_environment.observation_spec().shape

    def test_multi_env__action_spec(
        self,
        fake_multi_environment: FakeMultiEnvironment,
        fake_multi_to_single_env: MultiToSingleWrapper,
    ) -> None:
        """Validates action_spec property of the multi agent to single
        agent wrapped environment.
        """
        action_spec: specs.Array = fake_multi_to_single_env.action_spec()  # type: ignore
        assert isinstance(fake_multi_to_single_env.action_spec(), specs.Array)
        assert action_spec.shape == fake_multi_environment.action_spec().shape

    def test_multi_env__unwrapped(
        self,
        fake_multi_environment: FakeMultiEnvironment,
        fake_multi_to_single_env: MultiToSingleWrapper,
    ) -> None:
        """Validates unwrapped property of the multi agent to single
        agent wrapped environment.
        """
        assert isinstance(fake_multi_to_single_env.unwrapped, Environment)
        assert fake_multi_to_single_env._env == fake_multi_environment


class TestVmapWrapper:
    @pytest.fixture
    def fake_vmap_environment(self, fake_environment: Environment) -> VmapWrapper:
        return VmapWrapper(fake_environment)

    @pytest.fixture
    def keys(self) -> random.PRNGKey:
        return random.split(random.PRNGKey(0), num=5)

    def test_vmap_wrapper__init(self, fake_environment: Environment) -> None:
        """Validates initialization of the vmap wrapper."""
        vmap_env = VmapWrapper(fake_environment)
        assert isinstance(vmap_env, Environment)

    def test_vmap_env__reset(
        self, fake_vmap_environment: VmapWrapper, keys: random.PRNGKey
    ) -> None:
        """Validates reset function and timestep type of the vmap wrapped environment."""
        _, timestep, _ = jax.jit(fake_vmap_environment.reset)(keys)

        assert isinstance(timestep, TimeStep)
        assert_trees_all_equal(timestep.step_type, StepType.FIRST)
        assert timestep.observation.shape[0] == keys.shape[0]
        assert timestep.reward.shape == (keys.shape[0],)
        assert timestep.discount.shape == (keys.shape[0],)

    def test_vmap_env__step(
        self, fake_vmap_environment: VmapWrapper, keys: random.PRNGKey
    ) -> None:
        """Validates step function of the vmap environment."""
        state, timestep, _ = fake_vmap_environment.reset(
            keys
        )  # type: Tuple[FakeState, TimeStep, Extra]
        action = jax.vmap(
            lambda _: fake_vmap_environment.action_spec().generate_value()
        )(keys)

        state, next_timestep, _ = jax.jit(fake_vmap_environment.step)(
            state, action
        )  # type: Tuple[FakeState, TimeStep, Extra]

        assert_trees_are_different(next_timestep, timestep)
        assert_trees_all_equal(next_timestep.reward, 0)

        assert next_timestep.reward.shape == (keys.shape[0],)
        assert next_timestep.discount.shape == (keys.shape[0],)
        assert next_timestep.observation.shape[0] == keys.shape[0]

    def test_vmap_env__unwrapped(
        self, fake_environment: Environment, fake_vmap_environment: VmapWrapper
    ) -> None:
        """Validates unwrapped property of the vmap environment."""
        assert isinstance(fake_vmap_environment.unwrapped, Environment)
        assert fake_vmap_environment._env == fake_environment


class TestBraxEnvToJumanjiEnvironment:
    """
    Test the BraxEnvToJumanjiEnvironment wrapper that transforms a Brax Env into an Environment
    format.
    """

    @pytest.fixture
    def fake_brax_env(self, time_limit: int = 10) -> BraxEnv:
        """Creates a trivial Brax Env meant for unit testing."""
        return brax.envs.create("fast", auto_reset=False, episode_length=time_limit)

    @pytest.fixture
    def jumanji_environment_from_brax(self, fake_brax_env: BraxEnv) -> Environment:
        """Instantiates an Environment wrapped from a Brax env."""
        return BraxToJumanjiWrapper(fake_brax_env)

    def test_brax_env_to_jumanji_environment__init(
        self, fake_brax_env: BraxEnv
    ) -> None:
        """Validates initialization of the wrapper."""
        environment = BraxToJumanjiWrapper(fake_brax_env)
        assert isinstance(environment, Environment)

    def test_brax_env_to_jumanji_environment__reset(
        self, jumanji_environment_from_brax: Environment
    ) -> None:
        """Validates (jitted) reset function and timestep type of the wrapped environment."""
        state, timestep, extra = jax.jit(jumanji_environment_from_brax.reset)(
            jax.random.PRNGKey(0)
        )
        assert isinstance(state, BraxState)
        assert isinstance(timestep, TimeStep)
        assert timestep.step_type == StepType.FIRST
        assert extra is None

    def test_brax_env_to_jumanji_environment__step(
        self, jumanji_environment_from_brax: Environment
    ) -> None:
        """Validates (jitted) step function of the wrapped environment."""
        state, timestep, _ = jumanji_environment_from_brax.reset(jax.random.PRNGKey(0))
        action = jumanji_environment_from_brax.action_spec().generate_value()
        next_state, next_timestep, _ = jax.jit(jumanji_environment_from_brax.step)(
            state, action
        )
        assert_trees_are_different(timestep, next_timestep)
        assert_trees_are_different(state, next_state)

    def test_brax_env_to_jumanji_environment__observation_spec(
        self, jumanji_environment_from_brax: Environment
    ) -> None:
        """Validates observation_spec property of the wrapped environment."""
        assert isinstance(jumanji_environment_from_brax.observation_spec(), specs.Array)

    def test_brax_env_to_jumanji_environment__action_spec(
        self, jumanji_environment_from_brax: Environment
    ) -> None:
        """Validates action_spec property of the wrapped environment."""
        assert isinstance(jumanji_environment_from_brax.action_spec(), specs.Array)

    def test_brax_env_to_jumanji_environment__unwrapped(
        self, jumanji_environment_from_brax: Environment
    ) -> None:
        """Validates unwrapped property of the wrapped environment."""
        assert isinstance(jumanji_environment_from_brax.unwrapped, BraxEnv)


class TestAutoResetWrapper:
    @pytest.fixture
    def fake_auto_reset_environment(
        self, fake_environment: Environment
    ) -> AutoResetWrapper:
        return AutoResetWrapper(fake_environment)

    @pytest.fixture
    def key(self) -> random.PRNGKey:
        return random.PRNGKey(0)

    @pytest.fixture
    def fake_state_and_timestep(
        self, fake_auto_reset_environment: AutoResetWrapper, key: random.PRNGKey
    ) -> Tuple[State, TimeStep]:
        state, timestep, _ = jax.jit(fake_auto_reset_environment.reset)(key)
        return state, timestep

    def test_auto_reset_wrapper__init(self, fake_environment: Environment) -> None:
        """Validates initialization of the AutoResetWrapper."""
        auto_reset_env = AutoResetWrapper(fake_environment)
        assert isinstance(auto_reset_env, Environment)

    def test_auto_reset_wrapper__auto_reset(
        self,
        fake_auto_reset_environment: AutoResetWrapper,
        fake_state_and_timestep: Tuple[State, TimeStep],
    ) -> None:
        """Validates the auto_reset function of the AutoResetWrapper."""
        state, timestep = fake_state_and_timestep
        _, reset_timestep = jax.jit(fake_auto_reset_environment.auto_reset)(
            state, timestep
        )
        assert_trees_all_equal(timestep.observation, reset_timestep.observation)

    def test_auto_reset_wrapper__step_no_reset(
        self, fake_auto_reset_environment: AutoResetWrapper, key: random.PRNGKey
    ) -> None:
        """Validates that step function of the AutoResetWrapper does not do an
        auto-reset when the terminal state is not reached.
        """
        state, first_timestep, _ = fake_auto_reset_environment.reset(
            key
        )  # type: Tuple[FakeState, TimeStep, Extra]

        # Generate an action
        action = fake_auto_reset_environment.action_spec().generate_value()

        state, timestep, _ = jax.jit(fake_auto_reset_environment.step)(
            state, action
        )  # type: Tuple[FakeState, TimeStep, Extra]

        assert timestep.step_type == StepType.MID
        assert_trees_are_different(timestep, first_timestep)
        assert_trees_all_equal(timestep.reward, 0)

    def test_auto_reset_wrapper__step_reset(
        self,
        fake_environment: Environment,
        fake_auto_reset_environment: AutoResetWrapper,
        key: random.PRNGKey,
    ) -> None:
        """Validates that the auto-reset is done correctly by the step function
        of the AutoResetWrapper when the terminal timestep is reached.
        """
        state, first_timestep, _extra = fake_auto_reset_environment.reset(
            key
        )  # type: Tuple[FakeState, TimeStep, Extra]

        fake_environment.time_limit = 5  # type: ignore

        # Loop across time_limit so auto-reset occurs
        for _ in range(fake_environment.time_limit):  # type: ignore
            action = fake_auto_reset_environment.action_spec().generate_value()
            state, timestep, _extr = jax.jit(fake_auto_reset_environment.step)(  # type: ignore
                state, action
            )  # type: Tuple[FakeState, TimeStep, Extra]

        assert timestep.step_type == first_timestep.step_type == StepType.FIRST
        assert_trees_all_equal(timestep.observation, first_timestep.observation)
