import dm_env
from dm_env import specs

from jumanji.jax.env import JaxEnv
from jumanji.jax.wrappers import DeepMindEnvWrapper
from jumanji.testing.fakes import make_fake_dm_env, make_fake_jax_env


class TestWrappers:
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
        action = self.fake_dm_env.action_spec.generate_value()
        next_timestep = self.fake_dm_env.step(action)
        assert next_timestep != timestep

    def test_dm_env__observation_spec(self) -> None:
        """Validates observation_spec property of the wrapped environment."""
        assert isinstance(self.fake_dm_env.observation_spec, specs.Array)

    def test_dm_env__action_spec(self) -> None:
        """Validates action_spec property of the wrapped environment."""
        assert isinstance(self.fake_dm_env.action_spec, specs.Array)

    def test_dm_env__unwrapped(self) -> None:
        """Validates unwrapped property of the wrapped environment."""
        assert isinstance(self.fake_dm_env.unwrapped, JaxEnv)
