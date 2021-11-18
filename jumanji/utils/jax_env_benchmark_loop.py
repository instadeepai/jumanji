from typing import Callable, Optional, Tuple

import haiku as hk
from chex import PRNGKey
from dm_env import specs
from jax import jit, random

from jumanji.jax.env import JaxEnv, State
from jumanji.jax.types import Action, TimeStep
from jumanji.utils.stopper import should_terminate
from jumanji.utils.timeit import TimeIt


class JaxEnvBenchmarkLoop:
    """Environment loop for JaxEnv, used for benchmarking. Inspired by acme.environment_loop
    (https://github.com/deepmind/acme/blob/master/acme/environment_loop.py).
    Compiles reset and step functions. This loop is for benchmark purposes, as it takes a random
    action between each step. The environment speed is being measured as the number of steps
    per second.
    """

    def __init__(
        self,
        environment: JaxEnv,
        seed: int = 0,
    ):
        """Environment loop used to benchmark the number of steps per second
        of an environment of type JaxEnv.

        Args:
            environment: JaxEnv to benchmark on.
            seed: random seed used for action selection and environment reset.

        """
        if not isinstance(environment, JaxEnv):
            raise TypeError(
                "environment must be of type JaxEnv, "
                f"got {environment} of type {type(environment)} instead."
            )
        self._environment = environment
        self._rng = hk.PRNGSequence(seed)
        self.step_fn: Callable[[State, Action], Tuple[State, TimeStep]] = jit(
            environment.step
        )
        self.reset_fn: Callable[[PRNGKey], Tuple[State, TimeStep]] = jit(
            environment.reset
        )
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
        state, _ = self.reset_fn(next(self._rng))  # type: ignore
        action = self._random_action()
        _ = self.step_fn(state, action)

    def _run_episode(self) -> int:
        """Runs one episode in the environment."""
        episode_steps = 0
        state, timestep = self.reset_fn(next(self._rng))  # type: ignore
        while not timestep.last():
            action = self._random_action()
            state, timestep = self.step_fn(state, action)
            episode_steps += 1
        return episode_steps

    def run(
        self,
        num_episodes: Optional[int] = None,
        num_steps: Optional[int] = None,
        ms: bool = False,
        print_: bool = True,
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

        with TimeIt(tag="Compilation", print_=print_) as timer:
            self._compile()
            # Tells the timer we run only 1 step
            timer.frames = 1
        episode_count, step_count = 0, 0
        while not should_terminate(episode_count, step_count, num_episodes, num_steps):
            with TimeIt(tag="Run", ms=ms, print_=print_) as timer:
                episode_length = self._run_episode()
                timer.frames = episode_length
            episode_count += 1
            step_count += episode_length
