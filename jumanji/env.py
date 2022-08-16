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

"""Abstract environment class"""

import abc
from typing import Any, Generic, Literal, Tuple, TypeVar

from chex import PRNGKey

from jumanji import specs
from jumanji.types import Action, Extra, TimeStep

State = TypeVar("State")


class Environment(abc.ABC, Generic[State]):
    """Environment written in Jax that differs from the gym API to make the step and
    reset functions jittable. The state contains all the dynamics and data needed to step
    the environment, no computation stored in attributes of self.
    The API is inspired by [brax](https://github.com/google/brax/blob/main/brax/envs/env.py).
    """

    def __repr__(self) -> str:
        return "Environment."

    @abc.abstractmethod
    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep, Extra]:
        """Resets the environment to an initial state.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: State object corresponding to the new state of the environment,
            timestep: TimeStep object corresponding the first timestep returned by the environment,
            extra: metrics, default to None.
        """

    @abc.abstractmethod
    def step(self, state: State, action: Action) -> Tuple[State, TimeStep, Extra]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the action to take.

        Returns:
            state: State object corresponding to the next state of the environment,
            timestep: TimeStep object corresponding the timestep returned by the environment,
            extra: metrics, default to None.
        """

    @abc.abstractmethod
    def observation_spec(self) -> specs.Spec:
        """Returns the observation spec.

        Returns:
            observation_spec: a NestedSpec tree of spec.
        """

    @abc.abstractmethod
    def action_spec(self) -> specs.Spec:
        """Returns the action spec.

        Returns:
            action_spec: a NestedSpec tree of spec.
        """

    def reward_spec(self) -> specs.Array:
        """Describes the reward returned by the environment. By default, this is assumed to be a
        single float.

        Returns:
            reward_spec: a `specs.Array` spec.
        """
        return specs.Array(shape=(), dtype=float, name="reward")

    def discount_spec(self) -> specs.BoundedArray:
        """Describes the discount returned by the environment. By default, this is assumed to be a
        single float between 0 and 1.

        Returns:
            discount_spec: a `specs.BoundedArray` spec.
        """
        return specs.BoundedArray(
            shape=(), dtype=float, minimum=0.0, maximum=1.0, name="discount"
        )

    @property
    def unwrapped(self) -> "Environment":
        return self

    def render(self, state: State) -> Any:
        """Render frames of the environment for a given state.

        Args:
            state: State object containing the current dynamics of the environment.
        """
        raise NotImplementedError("Render method not implemented for this environment.")

    def close(self) -> None:
        """Perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """

    def __enter__(self) -> "Environment":
        """Support with-statement for the environment."""
        return self

    def __exit__(self, *args: Any) -> Literal[False]:
        """Support with-statement for the environment."""
        self.close()
        # propagate exception
        return False


class Wrapper(Environment[State], Generic[State]):
    """Wraps the environment to allow modular transformations.
    Source: https://github.com/google/brax/blob/main/brax/envs/env.py#L72
    """

    def __init__(self, env: Environment):
        super().__init__()
        self._env = env

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._env)})"

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self._env, name)

    @property
    def unwrapped(self) -> Environment:
        """Returns the wrapped env."""
        return self._env.unwrapped

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep, Extra]:
        """Resets the environment to an initial state.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: State object corresponding to the new state of the environment,
            timestep: TimeStep object corresponding the first timestep returned by the environment,
            extra: metrics, default to None.
        """
        return self._env.reset(key)

    def step(self, state: State, action: Action) -> Tuple[State, TimeStep, Extra]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the action to take.

        Returns:
            state: State object corresponding to the next state of the environment,
            timestep: TimeStep object corresponding the timestep returned by the environment,
            extra: metrics, default to None.
        """
        return self._env.step(state, action)

    def observation_spec(self) -> specs.Spec:
        """Returns the observation spec."""
        return self._env.observation_spec()

    def action_spec(self) -> specs.Spec:
        """Returns the action spec."""
        return self._env.action_spec()

    def render(self, state: State) -> Any:
        """Compute render frames during initialisation of the environment.

        Args:
            state: State object containing the dynamics of the environment.
        """
        return self._env.render(state)

    def close(self) -> None:
        """Perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        return self._env.close()

    def __enter__(self) -> "Wrapper":
        """Support with-statement for the environment."""
        return self

    def __exit__(self, *args: Any) -> Literal[False]:
        """Support with-statement for the environment."""
        self.close()
        # propagate exception
        return False


def make_environment_spec(environment: Environment) -> specs.EnvironmentSpec:
    """Returns an `EnvironmentSpec` describing values used by an environment."""
    return specs.EnvironmentSpec(
        observations=environment.observation_spec(),
        actions=environment.action_spec(),
        rewards=environment.reward_spec(),
        discounts=environment.discount_spec(),
    )
