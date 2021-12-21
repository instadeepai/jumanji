from functools import partial
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

import dm_env
import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from dm_env import specs
from jax import lax, random

from jumanji.jax.env import JaxEnv
from jumanji.jax.specs import EnvironmentSpec
from jumanji.jax.types import Action, TimeStep, restart, termination, transition
from jumanji.jax.wrappers import DeepMindEnvWrapper
from validation.agents import Agent, TrainingState, Transition


@dataclass
class FakeState:
    key: PRNGKey
    step: int


class FakeJaxEnv(JaxEnv):
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

    def reset(self, key: PRNGKey) -> Tuple[FakeState, TimeStep]:
        """Resets the environment to an initial state: step number is 0.

        Args:
            key: random key used to reset the environment.

        Returns:
            state, timestep: Tuple[State, TimeStep] containing the new state of the environment,
                as well as the first timestep.
        """

        state = FakeState(key=key, step=0)
        observation = jnp.array(self.observation_spec().generate_value())
        timestep = restart(observation=observation)
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
                reward=jnp.zeros((), float),
                observation=jnp.zeros(self.observation_shape, float),
            ),
            lambda _: transition(
                reward=jnp.zeros((), float),
                observation=jnp.zeros(self.observation_shape, float),
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


class FakeAgent(Agent):
    """
    A fake agent that inherits from Agent, for testing purposes.
    """

    def __init__(self, action_spec: Optional[specs.BoundedArray] = None) -> None:
        self._action_spec = action_spec or specs.DiscreteArray(1, name="action")

    @staticmethod
    def init_training_state(key: PRNGKey) -> TrainingState:
        """Returns an initialized learning state."""
        training_state = TrainingState()
        return training_state

    def select_action(
        self, training_state: TrainingState, observation: Array, key: PRNGKey
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
