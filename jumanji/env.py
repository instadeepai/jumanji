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

from __future__ import annotations

import abc
from typing import Any, Generic, Tuple, TypeVar

import chex
from typing_extensions import Protocol

from jumanji import specs
from jumanji.types import TimeStep


class StateProtocol(Protocol):
    """Enforce that the State for every Environment must implement a key."""

    key: chex.PRNGKey


State = TypeVar("State", bound="StateProtocol")
ActionSpec = TypeVar("ActionSpec", bound=specs.Array)


class Environment(abc.ABC, Generic[State, ActionSpec]):
    """Environment written in Jax that differs from the gym API to make the step and
    reset functions jittable. The state contains all the dynamics and data needed to step
    the environment, no computation stored in attributes of self.
    The API is inspired by [brax](https://github.com/google/brax/blob/main/brax/envs/env.py).
    """

    def __repr__(self) -> str:
        return "Environment."

    def __init__(self) -> None:
        """Initialize environment."""
        self._observation_spec = self._make_observation_spec()
        self._action_spec = self._make_action_spec()
        self._reward_spec = self._make_reward_spec()
        self._discount_spec = self._make_discount_spec()

    @abc.abstractmethod
    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Resets the environment to an initial state.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: State object corresponding to the new state of the environment,
            timestep: TimeStep object corresponding the first timestep returned by the environment,
        """

    @abc.abstractmethod
    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the action to take.

        Returns:
            state: State object corresponding to the next state of the environment,
            timestep: TimeStep object corresponding the timestep returned by the environment,
        """

    @property
    def observation_spec(self) -> specs.Spec:
        """Returns the observation spec.

        Returns:
            observation_spec: a potentially nested `Spec` structure representing the observation.
        """
        return self._observation_spec

    @abc.abstractmethod
    def _make_observation_spec(self) -> specs.Spec:
        """Returns new observation spec.

        Returns:
            observation_spec: a potentially nested `Spec` structure representing the observation.
        """

    @property
    def action_spec(self) -> ActionSpec:
        """Returns the action spec.

        Returns:
            action_spec: a potentially nested `Spec` structure representing the action.
        """
        return self._action_spec

    @abc.abstractmethod
    def _make_action_spec(self) -> ActionSpec:
        """Returns new action spec.

        Returns:
            action_spec: a potentially nested `Spec` structure representing the action.
        """

    @property
    def reward_spec(self) -> specs.Array:
        """Returns the reward spec. By default, this is assumed to be a single float.

        Returns:
            reward_spec: a `specs.Array` spec.
        """
        return self._reward_spec

    def _make_reward_spec(self) -> specs.Array:
        """Returns new reward spec. By default, this is assumed to be a single float.

        Returns:
            reward_spec: a `specs.Array` spec.
        """
        return specs.Array(shape=(), dtype=float, name="reward")

    @property
    def discount_spec(self) -> specs.BoundedArray:
        """Returns the discount spec. By default, this is assumed to be a single float between 0 and 1.

        Returns:
            discount_spec: a `specs.BoundedArray` spec.
        """
        return self._discount_spec

    def _make_discount_spec(self) -> specs.BoundedArray:
        """Returns new discount spec. By default, this is assumed to be a single float between 0 and 1.

        Returns:
            discount_spec: a `specs.BoundedArray` spec.
        """
        return specs.BoundedArray(
            shape=(), dtype=float, minimum=0.0, maximum=1.0, name="discount"
        )

    @property
    def unwrapped(self) -> Environment:
        return self

    def render(self, state: State) -> Any:
        """Render frames of the environment for a given state.

        Args:
            state: State object containing the current dynamics of the environment.
        """
        raise NotImplementedError("Render method not implemented for this environment.")

    def close(self) -> None:
        """Perform any necessary cleanup."""

    def __enter__(self) -> Environment:
        return self

    def __exit__(self, *args: Any) -> None:
        """Calls :meth:`close()`."""
        self.close()
