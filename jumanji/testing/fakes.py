from functools import partial
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

import brax.envs
import dm_env
import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from dm_env import specs
from jax import lax, random

from jumanji.jax.env import JaxEnv
from jumanji.jax.specs import EnvironmentSpec
from jumanji.jax.types import Action, Extra, TimeStep, restart, termination, transition
from jumanji.jax.wrappers import JaxEnvToDeepMindEnv
from validation.agents import Agent, TrainingState, Transition


@dataclass
class FakeState:
    key: PRNGKey
    step: int


class FakeJaxEnv(JaxEnv[FakeState]):
    """
    A fake environment that inherits from JaxEnv, for testing purposes.
    """

    def __init__(
        self,
        time_limit: int = 10,
        observation_shape: Tuple = (),
        num_action_values: int = 1,
    ):
        """Initialize a fake jax environment.

        Args:
            time_limit: horizon of an episode.
            observation_shape: shape of the dummy observation.
            num_action_values: number of values in the bounded discrete action space.
        """
        self.time_limit = time_limit
        self.observation_shape = observation_shape
        self.num_action_values = num_action_values

    def observation_spec(self) -> specs.Array:
        """Returns the observation spec.

        Returns:
            observation_spec: a `dm_env.specs.Array` spec.
        """

        return specs.Array(
            shape=self.observation_shape, dtype=jnp.float32, name="observation"
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec.

        Returns:
            action_spec: a `dm_env.specs.Array` spec.
        """

        return specs.DiscreteArray(
            self.num_action_values, dtype=jnp.int32, name="action"
        )

    def reset(self, key: PRNGKey) -> Tuple[FakeState, TimeStep, Extra]:
        """Resets the environment to an initial state: step number is 0.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: State object corresponding to the new state of the environment,
            timestep: TimeStep object corresponding the first timestep returned by the environment,
            extra: metrics, default to None.
        """

        state = FakeState(key=key, step=0)
        observation = jnp.array(self.observation_spec().generate_value())
        timestep = restart(observation=observation)
        return state, timestep, None

    def step(
        self, state: FakeState, action: Action
    ) -> Tuple[FakeState, TimeStep, Extra]:
        """Steps into the environment by doing nothing but increasing the step number.

        Args:
            state: State containing a random key and a step number.
            action: array.

        Returns:
            state: State object corresponding to the next state of the environment,
            timestep: TimeStep object corresponding the timestep returned by the environment,
            extra: metrics, default to None.
        """
        key, _ = random.split(state.key)
        next_step = state.step + 1
        next_state = FakeState(key=key, step=next_step)
        timestep = lax.cond(
            next_step >= self.time_limit,
            lambda _: termination(
                reward=jnp.zeros((), float),
                observation=jnp.zeros(self.observation_shape, float),
            ),
            lambda _: transition(
                reward=jnp.zeros((), float),
                observation=jnp.zeros(self.observation_shape, float),
            ),
            None,
        )
        return next_state, timestep, None


class FakeMultiJaxEnv(JaxEnv[FakeState]):
    """
    A fake multi agent environment that inherits from JaxEnv, for testing purposes.
    """

    def __init__(
        self,
        num_agents: int = 5,
        observation_shape: Tuple = (5, 5),
        num_action_values: int = 1,
        reward_per_step: float = 1.0,
        time_limit: int = 10,
    ):
        """Initialize a fake multi agent jax environment.

        Args:
            num_agents : the number of agents present in the environment.
            observation_shape: shape of the dummy observation. The leading
                dimension should always be (num_agents, ...)
            num_action_values: number of values in the bounded discrete action space.
            reward_per_step: the reward given to each agent every timestep.
            time_limit: horizon of an episode.
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
            observation_spec: a `dm_env.specs.Array` spec.
        """

        return specs.Array(
            shape=self.observation_shape, dtype=jnp.float_, name="observation"
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec.

        Returns:
            action_spec: a `dm_env.specs.Array` spec.
        """

        return specs.BoundedArray(
            (self.num_agents,), jnp.int_, 0, self.num_action_values - 1
        )

    def reward_spec(self) -> specs.Array:
        """Returns the reward spec.

        Returns:
            reward_spec: a `dm_env.specs.Array` spec.
        """
        return specs.Array(shape=(self.num_agents,), dtype=jnp.float_, name="reward")

    def discount_spec(self) -> specs.Array:
        """Describes the discount returned by the environment.

        Returns:
            discount_spec: a `dm_env.specs.Array` spec.
        """
        return specs.BoundedArray(
            shape=(self.num_agents,),
            dtype=jnp.float_,
            minimum=0.0,
            maximum=1.0,
            name="discount",
        )

    def reset(self, key: PRNGKey) -> Tuple[FakeState, TimeStep, Extra]:
        """Resets the environment to an initial state: step number is 0.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: State object corresponding to the new state of the environment,
            timestep: TimeStep object corresponding the first timestep returned by the environment,
            extra: metrics, default to None.
        """

        state = FakeState(key=key, step=0)
        observation = jnp.zeros(self.observation_shape, float)
        timestep = restart(observation=observation, shape=(self.num_agents,))
        return state, timestep, None

    def step(
        self, state: FakeState, action: Action
    ) -> Tuple[FakeState, TimeStep, Extra]:
        """Steps into the environment by doing nothing but increasing the step number.

        Args:
            state: State containing a random key and a step number.
            action: array.

        Returns:
            state: State object corresponding to the next state of the environment,
            timestep: TimeStep object corresponding the timestep returned by the environment,
            extra: metrics, default to None.
        """
        key = random.split(state.key, 1).squeeze(0)
        next_step = state.step + 1
        next_state = FakeState(key=key, step=next_step)
        timestep = lax.cond(
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
        return next_state, timestep, None


"""
Some common functions and classes that are used in testing throughout jumanji.
"""


def make_fake_jax_env(time_limit: int = 10) -> FakeJaxEnv:
    """Creates a fake jax environment."""
    return FakeJaxEnv(time_limit=time_limit)


def make_fake_multi_jax_env(time_limit: int = 10) -> FakeMultiJaxEnv:
    """Creates a fake multi agent jax environment."""
    return FakeMultiJaxEnv(time_limit=time_limit)


def make_fake_dm_env(time_limit: int = 10) -> dm_env.Environment:
    """Creates a fake jax environment wrapped as a dm_env.Environment."""
    return JaxEnvToDeepMindEnv(FakeJaxEnv(time_limit=time_limit))


def make_fake_brax_env(time_limit: int = 10) -> brax.envs.Env:
    """Creates a trivial Brax Env meant for unit testing."""
    return brax.envs.create("fast", auto_reset=False, episode_length=time_limit)


class FakeAgent(Agent):
    """
    A fake agent that inherits from Agent, for testing purposes.
    """

    def __init__(self, action_spec: Optional[specs.BoundedArray] = None) -> None:
        self._action_spec = action_spec or specs.DiscreteArray(1, name="action")

    def init_training_state(self, key: PRNGKey) -> TrainingState:
        """Returns an initialized learning state."""
        training_state = TrainingState()
        return training_state

    def select_action(
        self,
        training_state: TrainingState,
        observation: Array,
        key: PRNGKey,
        extra: Extra = None,
    ) -> Action:
        """Returns an action, here returns 0."""
        action = jnp.array(self._action_spec.generate_value())
        return action

    def sgd_step(
        self, training_state: TrainingState, batch_traj: Transition
    ) -> Tuple[TrainingState, Dict]:
        """Simulates a sgd step without actually computing any loss."""
        metrics: Dict = {}
        return training_state, metrics


def make_fake_agent() -> FakeAgent:
    """Creates a fake agent."""
    return FakeAgent()


def fake_transition(env_spec: EnvironmentSpec) -> Transition:
    """Returns a fake transition in the environment. Shape: ()."""
    return Transition(
        observation=jnp.array(env_spec.observations.generate_value()),
        action=jnp.array(env_spec.actions.generate_value()),
        reward=jnp.array(env_spec.rewards.generate_value()),
        discount=jnp.array(env_spec.discounts.generate_value()),
        next_observation=jnp.array(env_spec.observations.generate_value()),
        extra=None,
    )


def fake_traj(env_spec: EnvironmentSpec, n_steps: int = 1) -> Transition:
    """Returns a fake trajectory (sequence of transitions) in the environment. Shape: (n_steps,)."""
    traj: Transition = jax.tree_map(
        partial(jnp.tile, reps=(n_steps,)),
        fake_transition(env_spec),
    )
    return traj


def batch_fake_traj(
    env_spec: EnvironmentSpec, n_steps: int = 1, batch_size: int = 1
) -> Transition:
    """Returns a fake batch of trajectories in the environment. Shape: (batch_size,n_steps)."""
    batch_traj: Transition = jax.tree_map(
        partial(
            jnp.tile,
            reps=(
                batch_size,
                n_steps,
            ),
        ),
        fake_transition(env_spec),
    )
    return batch_traj


class FakeMultiAgent(FakeAgent):
    """
    An agent to select random correctly sized actions for the multi agent case.
    For testing purposes.
    """

    def __init__(
        self, action_spec: Optional[specs.BoundedArray] = None, num_agents: int = 1
    ) -> None:
        super().__init__(action_spec)
        self._num_agents = num_agents

    def select_action(
        self,
        training_state: TrainingState,
        observation: Array,
        key: PRNGKey,
        extra: Extra = None,
    ) -> jnp.ndarray:
        """Randomly selects actions in the shape (num_agents,)."""
        action = random.randint(
            key,
            (self._num_agents,),
            self._action_spec.minimum,
            self._action_spec.maximum,
        )
        return action
