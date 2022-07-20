from functools import partial
from typing import Dict

import chex
import jax
import jax.numpy as jnp
from jax import random

from jumanji import specs
from jumanji.env import make_environment_spec
from jumanji.testing import pytrees
from jumanji.testing.fakes import FakeJaxEnv, fake_transition
from validation.agents import RandomAgent, TrainingState


class TestRandomAgent:
    fake_jax_env = FakeJaxEnv(num_action_values=5)
    random_agent = RandomAgent(fake_jax_env.action_spec())

    def test_random_agent__init_training_state(self) -> TrainingState:
        """Test and return the training state."""
        key = random.PRNGKey(0)
        training_state = self.random_agent.init_training_state(key)
        return training_state

    def test_random_agent__select_action(self) -> None:
        """Check that sampling two actions with different keys lead to different actions."""
        training_state = self.test_random_agent__init_training_state()
        fake_obs = self.fake_jax_env.observation_spec().generate_value()
        action_key_1, action_key_2 = random.split(random.PRNGKey(0))
        action_1 = self.random_agent.select_action(
            training_state, fake_obs, action_key_1
        )
        action_2 = self.random_agent.select_action(
            training_state, fake_obs, action_key_2
        )
        assert jnp.isfinite(action_1).all()
        chex.assert_shape(action_1, self.fake_jax_env.action_spec().shape)
        chex.assert_type(action_1, self.fake_jax_env.action_spec().dtype)
        assert action_2 != action_1

    def test_random_agent__different_action_spaces(self) -> None:
        """Test action selection for a continuous and discrete action space, check correct type."""
        c_action_spec = specs.BoundedArray((5,), jnp.float32, 0, 1)
        d_action_spec = specs.BoundedArray((5,), jnp.int32, 0, 1)

        c_random_agent = RandomAgent(c_action_spec)
        d_random_agent = RandomAgent(d_action_spec)

        c_key = random.PRNGKey(0)
        d_key = random.PRNGKey(0)

        c_training_state = c_random_agent.init_training_state(c_key)
        d_training_state = c_random_agent.init_training_state(d_key)

        fake_obs = self.fake_jax_env.observation_spec().generate_value()
        c_action_key, d_action_key = random.split(random.PRNGKey(0))

        c_action = c_random_agent.select_action(
            c_training_state, fake_obs, c_action_key
        )
        d_action = d_random_agent.select_action(
            d_training_state, fake_obs, d_action_key
        )
        assert jnp.issubdtype(c_action.dtype, jnp.float32)
        assert jnp.issubdtype(d_action.dtype, jnp.int32)

    def test_random_agent__sgd_step(self) -> None:
        """Check the sgd step, here no learning is expected."""
        training_state = self.test_random_agent__init_training_state()
        batch_fake_traj = jax.tree_map(
            partial(jnp.expand_dims, axis=0),
            fake_transition(make_environment_spec(self.fake_jax_env)),
        )
        new_training_state, metrics = self.random_agent.sgd_step(
            training_state, batch_fake_traj
        )
        assert isinstance(metrics, Dict)
        assert type(training_state) is TrainingState
        assert type(new_training_state) is TrainingState
        pytrees.assert_trees_are_equal(training_state, new_training_state)
