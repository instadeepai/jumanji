from typing import Callable, Tuple

import haiku as hk
from chex import PRNGKey
from dm_env import specs
from jax import jit

from jumanji.jax.env import JaxEnv
from jumanji.jax.types import Action, Extra, State, TimeStep
from validation.benchmark_loops.benchmark_loop import BenchmarkLoop


class JaxEnvBenchmarkLoop(BenchmarkLoop):
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
        self.step_fn: Callable[[State, Action], Tuple[State, TimeStep, Extra]] = jit(
            environment.step
        )
        self.reset_fn: Callable[[PRNGKey], Tuple[State, TimeStep, Extra]] = jit(
            environment.reset
        )
        if not isinstance(self._environment.action_spec(), specs.BoundedArray):
            action_spec = self._environment.action_spec()
            raise TypeError(
                f"action spec must be of type BoundedArray, got "
                f"{action_spec} of type {type(action_spec)}."
            )

    def _compile(self) -> int:
        """Calls jittable functions reset_fn and step_fn for the first time to
        compile them under jit. Returns the number of steps taken during compilation.
        """
        state, *_ = self.reset_fn(next(self._rng))  # type: ignore
        action = self._random_action(
            key=next(self._rng), action_spec=self._environment.action_spec()
        )
        _ = self.step_fn(state, action)
        steps_taken = 1  # Only one step was taken in the environment.
        return steps_taken

    def _run_episode(self) -> int:
        """Runs one episode in the environment."""
        episode_steps = 0
        state, timestep, _ = self.reset_fn(next(self._rng))  # type: ignore
        while not timestep.last():
            action = self._random_action(
                key=next(self._rng), action_spec=self._environment.action_spec()
            )
            state, timestep, _ = self.step_fn(state, action)
            episode_steps += 1
        return episode_steps
