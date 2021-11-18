from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

import dm_env
import jax.numpy as jnp
from chex import PRNGKey
from dm_env import specs
from jax import lax, random

from jumanji.jax.env import JaxEnv
from jumanji.jax.types import Action, TimeStep, restart, termination, transition
from jumanji.jax.wrappers import DeepMindEnvWrapper


@dataclass
class FakeState:
    key: PRNGKey
    step: int


class FakeJaxEnv(JaxEnv):
    """
    A fake environment that inherits from JaxEnv.
    """

    def __init__(self, time_limit: int = 10):
        self.time_limit = time_limit

    @property
    def observation_spec(self) -> specs.Array:
        """Returns the observation spec. The observation is just a (useless) scalar here.

        Returns:
            observation_spec: dm_env.specs object
        """

        return specs.Array(shape=(), dtype=jnp.float32, name="observation")

    @property
    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec. The action is just one (useless) integer here.

        Returns:
            action_spec: dm_env.specs object
        """

        return specs.DiscreteArray(1, dtype=jnp.int32, name="action")

    def reset(self, key: PRNGKey) -> Tuple[FakeState, TimeStep]:
        """Resets the environment to an initial state: step number is 0.

        Args:
            key: random key used to reset the environment.

        Returns:
            state, timestep: Tuple[State, TimeStep] containing the new state of the environment,
                as well as the first timestep.
        """

        state = FakeState(key=key, step=0)
        timestep = restart(observation=jnp.zeros((), float))
        return state, timestep

    def step(self, state: FakeState, action: Action) -> Tuple[FakeState, TimeStep]:
        """Steps into the environment by doing nothing but increasing the step number.

        Args:
            state: State containing a random key and a step number.
            action: array.

        Returns:
            state, timestep: Tuple[State, TimeStep] containing the next state of the environment,
                as well as the timestep to be observed.
        """
        key, _ = random.split(state.key)
        next_step = state.step + 1
        next_state = FakeState(key=key, step=next_step)
        timestep = lax.cond(
            next_step >= self.time_limit,
            lambda _: termination(
                reward=jnp.zeros((), float), observation=jnp.zeros((), float)
            ),
            lambda _: transition(
                reward=jnp.zeros((), float), observation=jnp.zeros((), float)
            ),
            None,
        )
        return next_state, timestep


"""
Some common functions and classes that are used in testing throughout jumanji.
"""


def make_fake_jax_env(time_limit: int = 10) -> FakeJaxEnv:
    """Creates a fake jax environment."""
    return FakeJaxEnv(time_limit=time_limit)


def make_fake_dm_env(time_limit: int = 10) -> dm_env.Environment:
    """Creates a fake jax environment wrapped as a dm_env.Environment."""
    return DeepMindEnvWrapper(FakeJaxEnv(time_limit=time_limit))
