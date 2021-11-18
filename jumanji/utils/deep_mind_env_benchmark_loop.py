from typing import Callable, Optional

import dm_env
import haiku as hk
from dm_env import specs
from jax import random

from jumanji.jax.types import Action
from jumanji.utils.stopper import should_terminate
from jumanji.utils.timeit import TimeIt


class DeepMindEnvBenchmarkLoop:
    """Environment loop for dm_env.Environment. Inspired by acme.environment_loop
    (https://github.com/deepmind/acme/blob/master/acme/environment_loop.py).
    This loop is for benchmark purposes, as it takes a random action between each step. The
    environment speed is being measured as the number of steps per second.
    """

    def __init__(
        self,
        environment: dm_env.Environment,
        seed: int = 0,
    ):
        """Environment loop used to benchmark the number of steps per second
        of a dm_env.Environment.

        Args:
            environment: dm_env.Environment to benchmark on.
            seed: random seed used for action selection.

        """
        if not isinstance(environment, dm_env.Environment):
            raise TypeError(
                "environment must be of type dm_env.Environment, "
                f"got {environment} of type {type(environment)} instead."
            )
        self._environment = environment
        self._rng = hk.PRNGSequence(seed)
        self.step_fn: Callable[[Action], dm_env.TimeStep] = environment.step
        self.reset_fn: Callable[[], dm_env.TimeStep] = environment.reset
        if not isinstance(self._environment.action_spec, specs.BoundedArray):
            action_spec = self._environment.action_spec
            raise TypeError(
                f"action spec must be of type BoundedArray, got "
                f"{action_spec} of type {type(action_spec)}."
            )

    def _random_action(self) -> Action:
        """Select a random action for a specs.BoundedArray action space.

        Returns:
            action randomly selected in the discrete interval given by specs.BoundedArray.

        """
        action = random.randint(
            next(self._rng),
            self._environment.action_spec.shape,
            0,
            self._environment.action_spec._num_values,
        )
        return action

    def _compile(self) -> None:
        """Calls jittable functions reset_fn and step_fn for the first time to
        compile them under jit.
        """
        _ = self.reset_fn()
        action = self._random_action()
        _ = self.step_fn(action)

    def _run_episode(self) -> int:
        """Runs one episode in the environment."""
        episode_steps = 0
        timestep = self.reset_fn()
        while not timestep.last():
            action = self._random_action()
            timestep = self.step_fn(action)
            episode_steps += 1
        return episode_steps

    def run(
        self,
        num_episodes: Optional[int] = None,
        num_steps: Optional[int] = None,
        print_: bool = True,
        ms: bool = False,
    ) -> None:
        """Runs the environment loop for a certain number of steps or episodes.
        Actions are selected randomly for benchmarking purposes.

        Args:
            num_episodes: number of episodes to play in the environment.
            num_steps: number of steps to take in the environment (either num_episodes or
                num_steps should be None).
            print_: True to print, False to not print but log with info level.
            ms: True to have time displayed in milliseconds, False to have it in seconds.

        """
        if not (num_episodes is None or num_steps is None):
            raise ValueError('Either "num_episodes" or "num_steps" should be None.')

        with TimeIt(tag="Compilation", print_=print_):
            self._compile()
        episode_count, step_count = 0, 0
        while not should_terminate(episode_count, step_count, num_episodes, num_steps):
            with TimeIt(tag="Run", ms=ms, print_=print_) as timer:
                episode_length = self._run_episode()
                timer.frames = episode_length
            episode_count += 1
            step_count += episode_length
