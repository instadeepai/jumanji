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

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex
import jax
import jax.numpy as jnp

from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep, restart, termination, transition


@dataclass
class FakeState:
    key: chex.PRNGKey
    step: jnp.int32


class FakeEnvironment(Environment[FakeState]):
    """
    A fake environment that inherits from Environment, for testing purposes.
    The observation is an array full of `state.step` of shape `(self.observation_shape,)`
    """

    def __init__(
        self,
        time_limit: int = 10,
        observation_shape: chex.Shape = (),
        action_shape: chex.Shape = (2,),
    ):
        """Initialize a fake environment.

        Args:
            time_limit: time_limit of an episode.
            observation_shape: shape of the dummy observation.
            action_shape: shape of bounded continuous action space.
        """
        self.time_limit = time_limit
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self._example_action = self.action_spec().generate_value()

    def observation_spec(self) -> specs.Array:
        """Returns the observation spec.

        Returns:
            observation_spec: a `specs.Array` spec.
        """

        return specs.Array(
            shape=self.observation_shape, dtype=float, name="observation"
        )

    def action_spec(self) -> specs.BoundedArray:
        """Returns the action spec.

        Returns:
            action_spec: a `specs.DiscreteArray` spec.
        """

        return specs.BoundedArray(
            shape=self.action_shape,
            minimum=jnp.zeros(self.action_shape, float),
            maximum=jnp.ones(self.action_shape, float),
            dtype=float,
            name="action",
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[FakeState, TimeStep]:
        """Resets the environment to an initial state: step number is 0.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: State object corresponding to the new state of the environment,
            timestep: TimeStep object corresponding the first timestep returned by the environment,
        """

        state = FakeState(key=key, step=jnp.array(0, jnp.int32))
        observation = self._state_to_obs(state)
        timestep = restart(observation=observation)
        return state, timestep

    def step(self, state: FakeState, action: chex.Array) -> Tuple[FakeState, TimeStep]:
        """Steps into the environment by doing nothing but increasing the step number.

        Args:
            state: State containing a random key and a step number.
            action: array.

        Returns:
            state: State object corresponding to the next state of the environment,
            timestep: TimeStep object corresponding the timestep returned by the environment,
        """
        chex.assert_equal_shape((action, self._example_action))
        key, _ = jax.random.split(state.key)
        next_step = state.step + 1
        next_state = FakeState(key=key, step=next_step)
        observation = self._state_to_obs(next_state)
        timestep = jax.lax.cond(
            next_step >= self.time_limit,
            termination,
            transition,
            jnp.zeros((), float),
            observation,
        )
        return next_state, timestep

    def render(self, state: FakeState) -> Tuple[chex.Shape, chex.Shape]:
        """Render the state attributes as a tuple.

        Args:
            state: State object containing the current dynamics of the environment.

        Returns:
            A tuple of key and step shapes from the environment state.

        """
        return state.key.shape, state.step.shape

    def _state_to_obs(self, state: FakeState) -> chex.Array:
        """The observation is an array full of `state.step` of shape `(self.observation_shape,)`."""
        return state.step * jnp.ones(self.observation_shape, float)


class FakeMultiEnvironment(Environment[FakeState]):
    """
    A fake multi agent environment that inherits from Environment, for testing purposes.
    """

    def __init__(
        self,
        num_agents: int = 5,
        observation_shape: Tuple = (5, 5),
        num_action_values: int = 1,
        reward_per_step: float = 1.0,
        time_limit: int = 10,
    ):
        """Initialize a fake multi agent environment.

        Args:
            num_agents : the number of agents present in the environment.
            observation_shape: shape of the dummy observation. The leading
                dimension should always be (num_agents, ...)
            num_action_values: number of values in the bounded discrete action space.
            reward_per_step: the reward given to each agent every timestep.
            time_limit: time_limit of an episode.
        """
        self.time_limit = time_limit
        self.observation_shape = observation_shape
        self.num_action_values = num_action_values
        self.num_agents = num_agents
        self.reward_per_step = reward_per_step
        assert (
            observation_shape[0] == num_agents
        ), f"""a leading dimension of size 'num_agents': {num_agents} is expected
            for the observation, got shape: {observation_shape}."""

    def observation_spec(self) -> specs.Array:
        """Returns the observation spec.

        Returns:
            observation_spec: a `specs.Array` spec.
        """

        return specs.Array(
            shape=self.observation_shape, dtype=float, name="observation"
        )

    def action_spec(self) -> specs.BoundedArray:
        """Returns the action spec.

        Returns:
            action_spec: a `specs.Array` spec.
        """

        return specs.BoundedArray(
            (self.num_agents,), int, 0, self.num_action_values - 1
        )

    def reward_spec(self) -> specs.Array:
        """Returns the reward spec.

        Returns:
            reward_spec: a `specs.Array` spec.
        """
        return specs.Array(shape=(self.num_agents,), dtype=float, name="reward")

    def discount_spec(self) -> specs.BoundedArray:
        """Describes the discount returned by the environment.

        Returns:
            discount_spec: a `specs.BoundedArray` spec.
        """
        return specs.BoundedArray(
            shape=(self.num_agents,),
            dtype=float,
            minimum=0.0,
            maximum=1.0,
            name="discount",
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[FakeState, TimeStep]:
        """Resets the environment to an initial state: step number is 0.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: State object corresponding to the new state of the environment,
            timestep: TimeStep object corresponding the first timestep returned by the environment,
        """

        state = FakeState(key=key, step=0)
        observation = self.observation_spec().generate_value()
        timestep = restart(observation=observation, shape=(self.num_agents,))
        return state, timestep

    def step(self, state: FakeState, action: chex.Array) -> Tuple[FakeState, TimeStep]:
        """Steps into the environment by doing nothing but increasing the step number.

        Args:
            state: State containing a random key and a step number.
            action: array.

        Returns:
            state: State object corresponding to the next state of the environment,
            timestep: TimeStep object corresponding the timestep returned by the environment,
        """
        key = jax.random.split(state.key, 1).squeeze(0)
        next_step = state.step + 1
        next_state = FakeState(key=key, step=next_step)
        timestep = jax.lax.cond(
            next_step >= self.time_limit,
            lambda _: termination(
                reward=jnp.ones(self.num_agents, float) * self.reward_per_step,
                observation=jnp.zeros(self.observation_shape, float),
                shape=(self.num_agents,),
            ),
            lambda _: transition(
                reward=jnp.ones(self.num_agents, float) * self.reward_per_step,
                observation=jnp.zeros(self.observation_shape, float),
                shape=(self.num_agents,),
            ),
            None,
        )
        return next_state, timestep
