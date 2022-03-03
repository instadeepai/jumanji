import dm_env
import pytest
from dm_env import specs

from jumanji.pcb_grid.pcb_grid import PcbGridEnv
from jumanji.pcb_grid.wrappers import DeepMindEnvWrapper


class TestWrappers:
    env = PcbGridEnv(8, 8, 3, "easy")
    wrapped = DeepMindEnvWrapper(env, horizon=2)

    def test_dm_env_wrapper__init(self) -> None:
        """Validates initialization of the dm_env wrapper."""
        dm_environment = DeepMindEnvWrapper(self.env, horizon=2)
        assert isinstance(dm_environment, dm_env.Environment)
        assert dm_environment._step == 0
        assert dm_environment.horizon == 2

        dm_environment = DeepMindEnvWrapper(self.env)
        assert dm_environment.horizon == (8 + 8) * 3

        with pytest.raises(AssertionError):
            DeepMindEnvWrapper("Not PcbGridEnv")  # type: ignore

    def test_dm_env__reset(self) -> None:
        """Validates reset function and timestep type of the wrapped environment."""
        timestep = self.wrapped.reset()

        assert isinstance(timestep, dm_env.TimeStep)

        assert self.wrapped._step == 0

        assert timestep.observation.shape == self.wrapped.observation_spec().shape
        assert timestep.step_type is dm_env.StepType.FIRST
        assert timestep.reward is None
        assert timestep.discount is None

    def test_dm_env__step(self) -> None:
        """Validates step function of the wrapped environment."""
        timestep = self.wrapped.reset()
        action = self.wrapped.action_spec().generate_value()
        next_timestep = self.wrapped.step(action)

        assert next_timestep != timestep

        assert next_timestep.observation.shape == self.wrapped.observation_spec().shape
        # not sure how to test this
        assert isinstance(next_timestep.reward, float)
        assert isinstance(next_timestep.discount, float)

        assert next_timestep.step_type == dm_env.StepType.MID
        assert next_timestep.discount == 1
        assert self.wrapped._step == 1

        next_timestep = self.wrapped.step(action)
        assert next_timestep.step_type == dm_env.StepType.LAST
        assert next_timestep.discount == 0
        assert self.wrapped._step == 2

    def test_dm_env__observation_spec(self) -> None:
        """Validates observation_spec property of the wrapped environment."""
        assert isinstance(self.wrapped.observation_spec(), specs.Array)
        assert self.wrapped.observation_spec().shape == (
            self.wrapped._env.rows,
            self.wrapped._env.cols,
        )

    def test_dm_env__action_spec(self) -> None:
        """Validates action_spec property of the wrapped environment."""
        assert isinstance(self.wrapped.action_spec(), specs.Array)
        assert self.wrapped.action_spec().shape == (2,)

    def test_dm_env__discount_spec(self) -> None:
        """Validates discount_spec property of the wrapped environment."""
        assert isinstance(self.wrapped.discount_spec(), specs.BoundedArray)
        assert self.wrapped.discount_spec().shape == ()

    def test_dm_env__reward_spec(self) -> None:
        """Validates reward_spec property of the wrapped environment."""
        assert isinstance(self.wrapped.reward_spec(), specs.Array)
        assert self.wrapped.reward_spec().shape == ()

    def test_dm_env__unwrapped(self) -> None:
        """Validates unwrapped property of the wrapped environment."""
        assert isinstance(self.wrapped.unwrapped(), PcbGridEnv)

    def test_dm_env__episode_terminates(self) -> None:
        """Checks that an episode terminates."""
        timestep = self.wrapped.reset()
        while not timestep.last():
            action = self.wrapped.action_spec().generate_value()
            timestep = self.wrapped.step(action)
