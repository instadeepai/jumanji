"""Abstract environment class"""

import abc
from typing import Generic, Tuple, TypeVar

from chex import PRNGKey
from dm_env import specs

from jumanji.jax.types import Action, TimeStep

State = TypeVar("State")


class JaxEnv(abc.ABC, Generic[State]):
    """Environment written in Jax that differs from the gym API to make the step and
    reset functions jittable. The state contains all the dynamics and data needed to step
    the environment, no computation stored in attributes of self.
    The API is inspired by [brax](https://github.com/google/brax/blob/main/brax/envs/env.py).
    """

    def __repr__(self) -> str:
        return "Jax environment."

    @abc.abstractmethod
    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep]:
        """Resets the environment to an initial state.

        Args:
            key: random key used to reset the environment.

        Returns:
            state, timestep: Tuple[State, TimeStep] containing the new state of the environment,
                as well as the first timestep.
        """

    @abc.abstractmethod
    def step(self, state: State, action: Action) -> Tuple[State, TimeStep]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the action to take.

        Returns:
            state, timestep: Tuple[State, TimeStep] containing the next state of the environment,
                as well as the timestep to be observed.
        """

    @abc.abstractmethod
    def observation_spec(self) -> specs.Array:
        """Returns the observation spec.

        Returns:
            observation_spec: a `dm_env.specs.Array` spec.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def action_spec(self) -> specs.Array:
        """Returns the action spec.

        Returns:
            action_spec: a `dm_env.specs.Array` spec.
        """
        raise NotImplementedError

    def reward_spec(self) -> specs.Array:
        """Describes the reward returned by the environment. By default, this is assumed to be a
        single float.

        Returns:
            reward_spec: a `dm_env.specs.Array` spec.
        """
        return specs.Array(shape=(), dtype=float, name="reward")

    def discount_spec(self) -> specs.Array:
        """Describes the discount returned by the environment. By default, this is assumed to be a
        single float between 0 and 1.

        Returns:
            discount_spec: a `dm_env.specs.Array` spec.
        """
        return specs.BoundedArray(
            shape=(), dtype=float, minimum=0.0, maximum=1.0, name="discount"
        )
