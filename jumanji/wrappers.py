from typing import Any, Callable, Optional, Tuple, TypeVar

import dm_env.specs
import jax
import jax.numpy as jnp
from brax.envs import Env as BraxEnv
from brax.envs import State as BraxState
from chex import PRNGKey
from jax import jit, random

from jumanji import specs
from jumanji.env import JaxEnv, Wrapper
from jumanji.types import Action, Extra, TimeStep, restart, termination, transition

State = TypeVar("State")


class JaxEnvToDeepMindEnv(dm_env.Environment):
    """A wrapper that converts JaxEnv to dm_env.Environment."""

    def __init__(self, env: JaxEnv, key: Optional[PRNGKey] = None):
        """Create the wrapped environment.

        Args:
            env: JaxEnv environment to wrap to a dm_env.Environment.
            key: optional key to initialize the JaxEnv environment with.
        """
        self._env = env
        if key is None:
            self._key = random.PRNGKey(0)
        else:
            self._key = key
        self._state: Any
        self._jitted_reset: Callable[[PRNGKey], Tuple[State, TimeStep, Extra]] = jit(
            self._env.reset
        )
        self._jitted_step: Callable[
            [State, Action], Tuple[State, TimeStep, Extra]
        ] = jit(self._env.step)

    def __repr__(self) -> str:
        return str(self._env.__repr__())

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
        self._state, timestep, _ = self._jitted_reset(reset_key)
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

        self._state, timestep, _ = self._jitted_step(self._state, action)
        return dm_env.TimeStep(**timestep)

    def observation_spec(self) -> dm_env.specs.Array:
        """Returns the dm_env observation spec."""
        return specs.jumanji_specs_to_dm_env_specs(self._env.observation_spec())

    def action_spec(self) -> dm_env.specs.Array:
        """Returns the dm_env action spec."""
        return specs.jumanji_specs_to_dm_env_specs(self._env.action_spec())

    @property
    def unwrapped(self) -> JaxEnv:
        return self._env


class MultiToSingleJaxEnv(Wrapper):
    """A wrapper that converts a multi-agent JaxEnv to a single-agent JaxEnv."""

    def __init__(
        self,
        env: JaxEnv,
        reward_aggregator: Callable = jnp.sum,
        discount_aggregator: Callable = jnp.max,
    ):
        """Create the wrapped environment.

        Args:
            env: JaxEnv environment to wrap to a dm_env.Environment.
            reward_aggregator: a function to aggregate all agents rewards
                into a single scalar value, e.g. sum.
            discount_aggregator: a function to aggregate all agents discounts
                into a single scalar value, e.g. max.
        """
        super().__init__(env)
        self._reward_aggregator = reward_aggregator
        self._discount_aggregator = discount_aggregator

    def _aggregate_timestep(self, timestep: TimeStep) -> TimeStep:
        """Apply the reward and discount aggregator to a multi-agent
            timestep object to create a new timestep object that consists
            of a scalar reward and discount value.

        Args:
            timestep: the multi agent timestep object.

        Return:
            a single agent compatible timestep object."""

        return TimeStep(
            step_type=timestep.step_type,
            observation=timestep.observation,
            reward=self._reward_aggregator(timestep.reward),
            discount=self._discount_aggregator(timestep.discount),
        )

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep, Extra]:
        """Resets the environment to an initial state.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: State object corresponding to the new state of the environment,
            timestep: TimeStep object corresponding the first timestep returned by the environment,
            extra: metrics, default to None.
        """
        state, timestep, extra = self._env.reset(key)
        timestep = self._aggregate_timestep(timestep)
        return state, timestep, extra

    def step(self, state: Any, action: Action) -> Tuple[State, TimeStep, Extra]:
        """Run one timestep of the environment's dynamics.

        The rewards are aggregated into a single value based on the given reward aggregator.
        The discount value is set to the largest discount of all the agents. This
        essentially means that if any single agent is alive, the discount value won't be zero.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the action to take.

        Returns:
            state: State object corresponding to the next state of the environment,
            timestep: TimeStep object corresponding the timestep returned by the environment,
            extra: metrics, default to None.
        """
        state, timestep, extra = self._env.step(state, action)
        timestep = self._aggregate_timestep(timestep)
        return state, timestep, extra


class VmapWrapper(Wrapper):
    """Vectorized Jax env."""

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep, Extra]:
        """Resets the environment to an initial state.

        The first dimension of the key will dictate the number of concurrent environments.

        To obtain a key with the right first dimension, you may call `jax.random.split` on key
        with the parameter `num` representing the number of concurrent environments.

        Args:
            key: random keys used to reset the environments where the first dimension is the number
                of desired environments.

        Returns:
            state: State object corresponding to the new state of the environments,
            timestep: TimeStep object corresponding the first timesteps returned by the
                environments,
            extra: metrics, default to None.
        """
        state, timestep, extra = jax.vmap(self._env.reset)(key)
        return state, timestep, extra

    def step(self, state: State, action: Action) -> Tuple[State, TimeStep, Extra]:
        """Run one timestep of the environment's dynamics.

        The first dimension of the state will dictate the number of concurrent environments.

        See `VmapWrapper.reset` for more details on how to get a state of concurrent
        environments.

        Args:
            state: State object containing the dynamics of the environments.
            action: Array containing the actions to take.

        Returns:
            state: State object corresponding to the next states of the environments,
            timestep: TimeStep object corresponding the timesteps returned by the environments,
            extra: metrics, default to None.
        """
        state, timestep, extra = jax.vmap(self._env.step)(state, action)
        return state, timestep, extra


class BraxEnvToJaxEnv(JaxEnv):
    """
    A wrapper that converts a Brax environment to a JaxEnv for standardisation, use with the
    JaxEnvironmentLoop and to augment the API (add timesteps, metrics...).
    """

    def __init__(self, brax_env: BraxEnv):
        """Creates the JaxEnv wrapper for Brax environments.

        Args:
            brax_env: Brax Env object that is not wrapped by a ResetWrapper
        """
        self._env = brax_env

    def reset(self, key: PRNGKey) -> Tuple[BraxState, TimeStep, Extra]:
        """Resets the environment to an initial state.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: Brax State object corresponding to the new state of the environment,
            timestep: TimeStep object corresponding the first timestep returned by the environment,
            extra: metrics, default to None.
        """
        state = self._env.reset(key)
        timestep = restart(observation=state.obs)
        extra = None
        return state, timestep, extra

    def step(self, state: BraxState, action: Action) -> Tuple[State, TimeStep, Extra]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: Brax State object containing the dynamics of the environment.
            action: Array containing the action to take.

        Returns:
            state: Brax State object corresponding to the next state of the environment,
            timestep: TimeStep object corresponding the timestep returned by the environment,
            extra: metrics, default to None.
        """
        state = self._env.step(state, action)
        timestep = jax.lax.cond(
            state.done,
            lambda _state: termination(reward=_state.reward, observation=_state.obs),
            lambda _state: transition(reward=_state.reward, observation=_state.obs),
            state,
        )
        extra = None
        return state, timestep, extra

    def observation_spec(self) -> specs.Array:
        """Returns the observation spec.

        Returns:
            observation_spec: a `specs.Array` spec.
        """
        return specs.Array(
            shape=(self._env.observation_size,),
            dtype=jnp.float_,
            name="observation",
        )

    def action_spec(self) -> specs.BoundedArray:
        """Returns the action spec.

        Returns:
            action_spec: a `specs.BoundedArray` spec.
        """
        return specs.BoundedArray(
            shape=(self._env.action_size,),
            dtype=jnp.float_,
            minimum=-1.0,
            maximum=1.0,
            name="action",
        )

    @property
    def unwrapped(self) -> BraxEnv:
        return self._env
