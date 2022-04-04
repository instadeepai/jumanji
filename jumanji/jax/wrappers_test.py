from typing import Tuple

import dm_env
import jax
import jax.numpy as jnp
import jax.random as random
import pytest
from chex import assert_trees_all_equal
from dm_env import specs

from jumanji.jax.env import JaxEnv
from jumanji.jax.types import Extra, StepType, TimeStep
from jumanji.jax.wrappers import DeepMindEnvWrapper, MultiToSingleJaxEnv, VmapWrapper
from jumanji.testing.fakes import (
    FakeJaxEnv,
    FakeState,
    make_fake_dm_env,
    make_fake_jax_env,
    make_fake_multi_jax_env,
)
from jumanji.testing.pytrees import assert_trees_are_different


class TestDeepMindEnvWrapper:
    fake_jax_env = make_fake_jax_env()
    fake_dm_env = make_fake_dm_env()

    def test_dm_env_wrapper__init(self) -> None:
        """Validates initialization of the dm_env wrapper."""
        dm_environment = DeepMindEnvWrapper(self.fake_jax_env)
        assert isinstance(dm_environment, dm_env.Environment)

    def test_dm_env__reset(self) -> None:
        """Validates reset function and timestep type of the wrapped environment."""
        timestep = self.fake_dm_env.reset()
        assert isinstance(timestep, dm_env.TimeStep)
        assert timestep.step_type is dm_env.StepType.FIRST

    def test_dm_env__step(self) -> None:
        """Validates step function of the wrapped environment."""
        timestep = self.fake_dm_env.reset()
        action = self.fake_dm_env.action_spec().generate_value()
        next_timestep = self.fake_dm_env.step(action)
        assert next_timestep != timestep

    def test_dm_env__observation_spec(self) -> None:
        """Validates observation_spec property of the wrapped environment."""
        assert isinstance(self.fake_dm_env.observation_spec(), specs.Array)

    def test_dm_env__action_spec(self) -> None:
        """Validates action_spec property of the wrapped environment."""
        assert isinstance(self.fake_dm_env.action_spec(), specs.Array)

    def test_dm_env__unwrapped(self) -> None:
        """Validates unwrapped property of the wrapped environment."""
        assert isinstance(self.fake_dm_env.unwrapped(), JaxEnv)


class TestMultiToSingleJaxEnv:
    fake_multi_jax_env = make_fake_multi_jax_env()
    fake_multi_to_single_env = MultiToSingleJaxEnv(fake_multi_jax_env)

    def test_multi_env_wrapper__init(self) -> None:
        """Validates initialization of the multi agent to single agent wrapper."""
        single_agent_env = MultiToSingleJaxEnv(self.fake_multi_jax_env)
        assert isinstance(single_agent_env, JaxEnv)

    def test_multi_env__reset(self) -> None:
        """Validates reset function and timestep type of the multi agent
        to single agent wrapped environment."""
        _, timestep, _ = self.fake_multi_to_single_env.reset(random.PRNGKey(0))
        assert isinstance(timestep, TimeStep)
        assert timestep.step_type is StepType.FIRST
        assert timestep.observation.shape[0] == self.fake_multi_jax_env.num_agents
        assert timestep.reward.shape == ()
        assert timestep.discount.shape == ()

    def test_multi_env__step(self) -> None:
        """Validates step function of the multi agent to single
        agent wrapped environment."""
        state, timestep, _ = self.fake_multi_to_single_env.reset(
            random.PRNGKey(0)
        )  # type: Tuple[FakeState, TimeStep, Extra]
        action = self.fake_multi_to_single_env.action_spec().generate_value()
        state, next_timestep, _ = self.fake_multi_to_single_env.step(
            state, action
        )  # type: Tuple[FakeState, TimeStep, Extra]
        assert next_timestep != timestep
        assert next_timestep.reward.shape == ()
        assert (
            next_timestep.reward
            == self.fake_multi_jax_env.reward_per_step
            * self.fake_multi_jax_env.num_agents
        )
        assert next_timestep.discount.shape == ()
        assert next_timestep.observation.shape[0] == self.fake_multi_jax_env.num_agents

    def test_multi_env__different_reward_aggregator(self) -> None:
        """Checks that using a different reward aggregator is correct"""
        mean_fake_multi_to_single_env = MultiToSingleJaxEnv(
            self.fake_multi_jax_env, reward_aggregator=jnp.mean
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
        assert next_timestep.reward == self.fake_multi_jax_env.reward_per_step
        assert next_timestep.discount.shape == ()
        assert next_timestep.observation.shape[0] == self.fake_multi_jax_env.num_agents

    def test_multi_env__observation_spec(self) -> None:
        """Validates observation_spec property of the multi agent to single
        agent wrapped environment."""
        assert isinstance(self.fake_multi_to_single_env.observation_spec(), specs.Array)
        assert (
            self.fake_multi_to_single_env.observation_spec()
            == self.fake_multi_jax_env.observation_spec()
        )

    def test_multi_env__action_spec(self) -> None:
        """Validates action_spec property of the multi agent to single
        agent wrapped environment."""
        assert isinstance(self.fake_multi_to_single_env.action_spec(), specs.Array)
        assert (
            self.fake_multi_to_single_env.action_spec()
            == self.fake_multi_jax_env.action_spec()
        )

    def test_multi_env__unwrapped(self) -> None:
        """Validates unwrapped property of the multi agent to single
        agent wrapped environment."""
        assert isinstance(self.fake_multi_to_single_env.unwrapped, JaxEnv)
        assert self.fake_multi_to_single_env._env == self.fake_multi_jax_env


class TestVmapWrapper:
    @pytest.fixture
    def fake_jax_env(self) -> FakeJaxEnv:
        return make_fake_jax_env()

    @pytest.fixture
    def fake_vmap_jax_env(self, fake_jax_env: JaxEnv) -> VmapWrapper:
        return VmapWrapper(fake_jax_env)

    @pytest.fixture
    def keys(self) -> random.PRNGKey:
        return random.split(random.PRNGKey(0), num=5)

    def test_vmap_wrapper__init(self, fake_jax_env: JaxEnv) -> None:
        """Validates initialization of the vmap wrapper."""
        vmap_env = VmapWrapper(fake_jax_env)
        assert isinstance(vmap_env, JaxEnv)

    def test_vmap_env__reset(
        self, fake_vmap_jax_env: VmapWrapper, keys: random.PRNGKey
    ) -> None:
        """Validates reset function and timestep type of the vmap wrapped environment."""
        _, timestep, _ = jax.jit(fake_vmap_jax_env.reset)(keys)

        assert isinstance(timestep, TimeStep)
        assert_trees_all_equal(timestep.step_type, StepType.FIRST)
        assert timestep.observation.shape[0] == keys.shape[0]
        assert timestep.reward.shape == (keys.shape[0],)
        assert timestep.discount.shape == (keys.shape[0],)

    def test_vmap_env__step(
        self, fake_vmap_jax_env: VmapWrapper, keys: random.PRNGKey
    ) -> None:
        """Validates step function of the vmap environment."""
        state, timestep, _ = fake_vmap_jax_env.reset(
            keys
        )  # type: Tuple[FakeState, TimeStep, Extra]
        action = jax.vmap(lambda _: fake_vmap_jax_env.action_spec().generate_value())(
            keys
        )

        state, next_timestep, _ = jax.jit(fake_vmap_jax_env.step)(
            state, action
        )  # type: Tuple[FakeState, TimeStep, Extra]

        assert_trees_are_different(next_timestep, timestep)
        assert_trees_all_equal(next_timestep.reward, 0)

        assert next_timestep.reward.shape == (keys.shape[0],)
        assert next_timestep.discount.shape == (keys.shape[0],)
        assert next_timestep.observation.shape[0] == keys.shape[0]

    def test_vmap_env__unwrapped(
        self, fake_jax_env: JaxEnv, fake_vmap_jax_env: VmapWrapper
    ) -> None:
        """Validates unwrapped property of the vmap environment."""
        assert isinstance(fake_vmap_jax_env.unwrapped, JaxEnv)
        assert fake_vmap_jax_env._env == fake_jax_env
