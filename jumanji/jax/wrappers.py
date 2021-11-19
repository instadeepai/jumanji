from typing import Any, Callable, Optional, Tuple

import dm_env
from chex import PRNGKey
from dm_env import specs
from jax import jit, random

from jumanji.jax.env import JaxEnv, State
from jumanji.jax.types import Action, TimeStep


class DeepMindEnvWrapper(dm_env.Environment):
    """A wrapper that converts JaxEnv to dm_env.Environment."""

    def __init__(self, env: JaxEnv, key: Optional[PRNGKey] = None):
        """Create the wrapped environment.

        Args:
            env: JaxEnv environment to wrap to a dm_env.Environment.
            key: optional key to initialize the JaxEnv environment with.
        """
        self._env = env
        self._key = key or random.PRNGKey(0)
        self._state: Any
        self._jitted_reset: Callable[[PRNGKey], Tuple[State, TimeStep]] = jit(
            self._env.reset
        )
        self._jitted_step: Callable[[State, Action], Tuple[State, TimeStep]] = jit(
            self._env.step
        )

    def reset(self) -> dm_env.TimeStep:
        """Starts a new sequence and returns the first `TimeStep` of this sequence.

        Returns:
            A `TimeStep` namedtuple containing:
            step_type: A `StepType` of `FIRST`.
            reward: `None`, indicating the reward is undefined.
            discount: `None`, indicating the discount is undefined.
            observation: A NumPy array, or a nested dict, list or tuple of arrays.
                Scalar values that can be cast to NumPy arrays (e.g. Python floats)
                are also valid in place of a scalar array. Must conform to the
                specification returned by `observation_spec()`.
        """
        reset_key, self._key = random.split(self._key)
        self._state, timestep = self._jitted_reset(reset_key)
        return dm_env.restart(observation=timestep.observation)

    def step(self, action: Action) -> dm_env.TimeStep:
        """Updates the environment according to the action and returns a `TimeStep`.

        If the environment returned a `TimeStep` with `StepType.LAST` at the
        previous step, this call to `step` will start a new sequence and `action`
        will be ignored.

        This method will also start a new sequence if called after the environment
        has been constructed and `reset` has not been called. Again, in this case
        `action` will be ignored.

        Args:
            action: A NumPy array, or a nested dict, list or tuple of arrays
                corresponding to `action_spec()`.

        Returns:
            A `TimeStep` namedtuple containing:
                step_type: A `StepType` value.
                reward: Reward at this timestep, or None if step_type is
                    `StepType.FIRST`. Must conform to the specification returned by
                    `reward_spec()`.
                discount: A discount in the range [0, 1], or None if step_type is
                    `StepType.FIRST`. Must conform to the specification returned by
                    `discount_spec()`.
                observation: A NumPy array, or a nested dict, list or tuple of arrays.
                    Scalar values that can be cast to NumPy arrays (e.g. Python floats)
                    are also valid in place of a scalar array. Must conform to the
                    specification returned by `observation_spec()`.
        """

        self._state, timestep = self._jitted_step(self._state, action)
        return dm_env.TimeStep(**timestep)

    def observation_spec(self) -> specs.Array:
        """Returns the observation spec."""
        return self._env.observation_spec()

    def action_spec(self) -> specs.Array:
        """Returns the action spec."""
        return self._env.action_spec()

    def unwrapped(self) -> JaxEnv:
        return self._env
