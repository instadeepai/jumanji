from typing import Callable

import dm_env
import dm_env.specs
import haiku as hk

from jumanji.types import Action
from validation.benchmark_loops.benchmark_loop import BenchmarkLoop


class DeepMindEnvBenchmarkLoop(BenchmarkLoop):
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
        if not isinstance(self._environment.action_spec(), dm_env.specs.BoundedArray):
            action_spec = self._environment.action_spec()
            raise TypeError(
                f"action spec must be of type BoundedArray, got "
                f"{action_spec} of type {type(action_spec)}."
            )

    def _compile(self) -> int:
        """Calls jittable functions reset_fn and step_fn for the first time to
        compile them under jit. Returns the number of steps taken during compilation.
        """
        _ = self.reset_fn()
        action = self._random_action(
            key=next(self._rng), action_spec=self._environment.action_spec()
        )
        _ = self.step_fn(action)
        steps_taken = 1  # Only one step was taken in the environment.
        return steps_taken

    def _run_episode(self) -> int:
        """Runs one episode in the environment."""
        episode_steps = 0
        timestep = self.reset_fn()
        while not timestep.last():
            action = self._random_action(
                key=next(self._rng), action_spec=self._environment.action_spec()
            )
            timestep = self.step_fn(action)
            episode_steps += 1
        return episode_steps
