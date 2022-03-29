from functools import partial
from typing import Any, Callable, Dict, Generic, NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from chex import PRNGKey
from dm_env import specs
from jax import lax, random

from jumanji.jax.env import JaxEnv
from jumanji.jax.types import Action, Extra, State, TimeStep
from validation.agents import Agent, TrainingState, Transition
from validation.utils import loggers
from validation.utils.timeit import TimeIt


class ActingState(NamedTuple, Generic[State]):
    """Container for data used during the acting in the environment."""

    state: State
    timestep: TimeStep
    key: PRNGKey
    reset: jnp.bool_
    episode_count: jnp.int32
    extra: Extra


class JaxEnvironmentLoop:
    """Training loop designed for JaxEnv environments. It both acts in an environment on a batch
    of states and learns from them. The loop compiles and vmap sequences of steps.
    """

    def __init__(
        self,
        environment: JaxEnv,
        agent: Agent,
        n_steps: int = 1,
        batch_size: int = 1,
        seed: int = 0,
        logger: Optional[loggers.Logger] = None,
    ):
        """Environment loop used for JaxEnv environments.

        Args:
            environment: JaxEnv to train on.
            agent: RL agent that learns to maximize expected return in the given environment.
            n_steps: number of steps to execute in a sequence, usually 10-20.
            batch_size: number of different environment states to run and update
                in parallel.
            seed: random seed used for action selection and environment reset.

        """
        if not isinstance(environment, JaxEnv):
            raise TypeError(
                "environment must be of type JaxEnv, "
                f"got {environment} of type {type(environment)} instead."
            )
        self._environment = environment
        self._agent = agent
        self._n_steps = n_steps
        self._batch_size = batch_size
        self._steps_per_epoch = self._n_steps * self._batch_size
        self._rng = hk.PRNGSequence(seed)
        self._step_fn: Callable[
            [State, Action], Tuple[Any, TimeStep, Extra]
        ] = environment.step
        self._reset_fn: Callable[
            [PRNGKey], Tuple[Any, TimeStep, Extra]
        ] = environment.reset
        if not isinstance(environment.action_spec(), specs.BoundedArray):
            action_spec = environment.action_spec()
            raise TypeError(
                f"action spec must be of type BoundedArray, got "
                f"{action_spec} of type {type(action_spec)}."
            )
        self._logger = logger or loggers.TerminalLogger("train")

    @staticmethod
    def should_terminate(
        episode_count: int,
        step_count: int,
        num_episodes: Optional[int],
        num_steps: Optional[int],
    ) -> bool:
        """Checks whether the training should stop, depending on the number of
        episodes or steps run in the environment.

        Args:
            episode_count: current number of episodes run in the environment.
            step_count: current number of steps taken in the environment.
            num_episodes: number of episodes to play in the environment.
            num_steps: number of steps to take in the environment (either num_episodes or
                num_steps should be None).

        Returns:
            True if training should stop, else False.
        """
        return (num_episodes is not None and episode_count >= num_episodes) or (
            num_steps is not None and step_count >= num_steps
        )

    def _run_steps(
        self, acting_state: ActingState, training_state: TrainingState
    ) -> Tuple[ActingState, Transition]:
        """Runs a sequence of self._n_steps steps in the environment.

        Args:
            acting_state: ActingState namedtuple containing:
                - episode_count: current number of episodes finished during
                    the sequence.
                - key: random key used to reset or take an action.
                - reset: boolean informing whether to reset the state or not.
                - state: current State of the environment.
                - timestep: TimeStep containing the previous observation, reward, termination and
                    discount.
            training_state: current parameters, optimizer state and counter.

        Returns:
            next_acting_state: acting_state after taking n steps in the environment.
            timesteps: sequence of timesteps observed while taking steps.

        """

        def make_terminal_step(
            _acting_state: ActingState,
        ) -> Tuple[ActingState, Transition]:
            """Resets the environment because _acting_state['reset'] is True."""
            next_key, reset_key = random.split(_acting_state.key)
            timestep = _acting_state.timestep
            next_state, next_timestep, extra = self._reset_fn(reset_key)
            next_acting_state = ActingState(
                episode_count=_acting_state.episode_count + 1,
                key=next_key,
                reset=jnp.array(False, dtype=bool),
                state=next_state,
                timestep=next_timestep,
                extra=extra,
            )
            transition = Transition(
                observation=timestep.observation,
                action=-1
                * jnp.ones(
                    self._environment.action_spec().shape,
                    dtype=self._environment.action_spec().dtype,
                ),
                reward=timestep.reward,
                discount=timestep.discount,
                next_observation=next_timestep.observation,
                extra=extra,
            )
            return next_acting_state, transition

        def make_non_terminal_step(
            _acting_state: ActingState,
        ) -> Tuple[ActingState, Transition]:
            """Steps the environment because _acting_state['reset'] is False."""
            next_key, action_key = random.split(_acting_state.key)
            timestep = _acting_state.timestep
            action = self._agent.select_action(
                training_state,
                timestep.observation,
                action_key,
                extra=_acting_state.extra,
            )
            next_state, next_timestep, extra = self._step_fn(
                _acting_state.state, action
            )
            next_acting_state = ActingState(
                episode_count=_acting_state.episode_count,
                key=next_key,
                reset=next_timestep.last(),
                state=next_state,
                timestep=next_timestep,
                extra=extra,
            )
            transition = Transition(
                observation=timestep.observation,
                action=action,
                reward=timestep.reward,
                discount=timestep.discount,
                next_observation=next_timestep.observation,
                extra=extra,
            )
            return next_acting_state, transition

        def run_one_step(_acting_state: ActingState) -> Tuple[ActingState, Transition]:
            """Runs one step in the environment. The behavior is a function
            of _acting_state['reset'].

            Args:
                _acting_state: current acting_state for stepping in the environment.

            Returns:
                next_acting_state: updated acting state after running one step.
                transition: transition observed from taking the step.

            """
            next_acting_state, transition = lax.cond(
                _acting_state.reset,
                make_terminal_step,
                make_non_terminal_step,
                _acting_state,
            )
            return next_acting_state, transition

        acting_state, traj = lax.scan(
            lambda _acting_state, _: run_one_step(_acting_state),
            acting_state,
            xs=None,
            length=self._n_steps,
            unroll=1,
        )
        return acting_state, traj

    @partial(jax.jit, static_argnames=["self"])
    def run_epoch(
        self,
        acting_states: ActingState,
        training_state: TrainingState,
    ) -> Tuple[ActingState, Transition, TrainingState, Dict]:
        """Runs the acting in the environment to collect a batch of trajectories, and then takes
        a learning step.

        Args:
            acting_states: batch of acting states containing:
                - episode_count: current number of episodes finished during
                    the sequence.
                - key: random key used to reset or take an action.
                - reset: boolean informing whether to reset the state or not.
                - state: current State of the environment.
                - timestep: TimeStep containing the previous observation, reward, termination and
                    discount.
            training_state: current parameters, optimizer state and counter.

        Returns:
            acting_states: batch of acting states after taking environment steps.
            batch_traj: batch of trajectories collected during the epoch.
            training_state: updated parameters, optimizer state and counter.

        """
        acting_states, batch_traj = jax.vmap(self._run_steps, in_axes=(0, None))(
            acting_states, training_state
        )
        training_state, metrics = self._agent.sgd_step(training_state, batch_traj)
        metrics.update(
            reward=batch_traj.reward.sum(axis=-1).mean(axis=0),
        )
        return acting_states, batch_traj, training_state, metrics

    def run_and_time_epoch(
        self,
        acting_states: ActingState,
        training_state: TrainingState,
        episode_count: int,
        step_count: int,
        epoch_label: str = "train",
        ms: bool = False,
    ) -> Tuple[ActingState, TrainingState, int, int]:
        """Runs and times an epoch of acting and learning.

        Args:
            acting_states: batch of acting states after taking environment steps.
            training_state: current parameters, optimizer state and counter.
            episode_count: current number of episodes.
            step_count: current number of environment steps.
            epoch_label: str defining the computation done in the epoch: train or compilation.
            ms: True to have time displayed in milliseconds, False to have it in seconds.

        Returns:
            acting_states: batch of acting states after taking environment steps.
            training_state: updated parameters, optimizer state and counter.
            episode_count: updated number of episodes.
            step_count: updated number of environment steps.
        """
        self._logger.label = epoch_label
        with TimeIt(frames=self._steps_per_epoch, ms=ms, logger=self._logger):
            acting_states, batch_traj, training_state, metrics = self.run_epoch(
                acting_states, training_state
            )
        episode_count += acting_states.episode_count.sum()
        step_count += self._steps_per_epoch
        self._logger.write(metrics)
        return acting_states, training_state, episode_count, step_count

    def run(
        self,
        num_episodes: Optional[int] = None,
        num_steps: Optional[int] = None,
        ms: bool = False,
    ) -> None:
        """Runs the environment loop for a certain number of steps or episodes.
        Actions are selected randomly for benchmarking purposes.

        Args:
            num_episodes: number of episodes to play in the environment.
            num_steps: number of steps to take in the environment (either num_episodes or
                num_steps should be None).
            ms: True to have time displayed in milliseconds, False to have it in seconds.

        """
        if not (num_episodes is None or num_steps is None):
            raise ValueError('Either "num_episodes" or "num_steps" should be None.')

        episode_count, step_count = 0, 0
        training_state = self._agent.init_training_state(next(self._rng))
        keys = random.split(next(self._rng), self._batch_size)
        states, timesteps, extra = jax.vmap(self._reset_fn)(keys)
        acting_states = ActingState(
            episode_count=jnp.zeros((self._batch_size,), int),
            key=random.split(next(self._rng), self._batch_size),
            reset=jnp.zeros((self._batch_size,), bool),
            state=states,
            timestep=timesteps,
            extra=extra,
        )
        (
            acting_states,
            training_state,
            episode_count,
            step_count,
        ) = self.run_and_time_epoch(
            acting_states,
            training_state,
            episode_count,
            step_count,
            epoch_label="compilation",
            ms=ms,
        )
        while not self.should_terminate(
            episode_count, step_count, num_episodes, num_steps
        ):
            (
                acting_states,
                training_state,
                episode_count,
                step_count,
            ) = self.run_and_time_epoch(
                acting_states,
                training_state,
                episode_count,
                step_count,
                epoch_label="train",
                ms=ms,
            )
