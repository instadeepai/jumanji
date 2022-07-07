import abc
from typing import Optional, Union

import dm_env.specs
import numpy as np
from chex import PRNGKey
from jax import random

from jumanji.jax import specs
from jumanji.jax.types import Action
from validation.utils import loggers
from validation.utils.timeit import TimeIt


class BenchmarkLoop(abc.ABC):
    """Training loop that is only used for benchmarking purposes."""

    logger = loggers.TerminalLogger("benchmark")

    def __repr__(self) -> str:
        return "Environment loop."

    @staticmethod
    def should_terminate(
        episode_count: int,
        step_count: int,
        num_episodes: Optional[int],
        num_steps: Optional[int],
    ) -> bool:
        """Checks whether the training should stop depending on the number of
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

    def _random_action(
        self,
        key: PRNGKey,
        action_spec: Union[dm_env.specs.BoundedArray, specs.BoundedArray],
    ) -> Action:
        """Select a random action for a specs.BoundedArray action space.

        Returns:
            action randomly selected in the discrete interval given by specs.BoundedArray.

        """
        if np.issubdtype(action_spec.dtype, np.int32) or np.issubdtype(
            action_spec.dtype, np.int64
        ):
            action = random.randint(
                key=key,
                shape=action_spec.shape,
                minval=action_spec.minimum,
                maxval=action_spec.maximum + 1,
                dtype=action_spec.dtype,
            )
        elif np.issubdtype(action_spec.dtype, np.float32) or np.issubdtype(
            action_spec.dtype, np.float64
        ):
            action = random.uniform(
                key=key,
                shape=action_spec.shape,
                dtype=action_spec.dtype,
                minval=action_spec.minimum,
                maxval=action_spec.maximum,
            )
        else:
            raise ValueError(
                f"`action_spec.dtype` must be integral or float, got "
                f"{action_spec.dtype}."
            )
        return action

    @abc.abstractmethod
    def _compile(self) -> int:
        """Calls jittable functions reset_fn and step_fn for the first time to
        compile them under jit. Returns the number of steps taken during compilation.
        """

    @abc.abstractmethod
    def _run_episode(self) -> int:
        """Runs one episode in the environment."""

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

        with TimeIt(logger=self.logger) as timer:
            # Tells the timer how many steps we run to compile
            timer.frames = self._compile()
        episode_count, step_count = 0, 0
        while not self.should_terminate(
            episode_count, step_count, num_episodes, num_steps
        ):
            with TimeIt(logger=self.logger, ms=ms) as timer:
                episode_length = self._run_episode()
                timer.frames = episode_length
            episode_count += 1
            step_count += episode_length
