from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import haiku as hk
import jax.numpy as jnp
from chex import ArrayTree, PRNGKey
from dm_env import specs
from jax import jit, lax, random, vmap

from jumanji.jax.env import JaxEnv, State
from jumanji.jax.types import Action, TimeStep
from jumanji.utils.stopper import should_terminate
from jumanji.utils.timeit import TimeIt


class JaxEnvironmentLoop:
    """Example of an environment loop for a JaxEnv. Actions are taken randomly.
    Compiles and vmap sequences of steps. Inspired by acme.environment_loop
    (https://github.com/deepmind/acme/blob/master/acme/environment_loop.py).
    This loop is for benchmark purposes, as it takes a random action between each step. The
    environment speed is being measured as the number of steps per second.
    """

    def __init__(
        self,
        environment: JaxEnv,
        n_steps: int = 1,
        batch_size: int = 1,
        seed: int = 0,
    ):
        """Environment loop used to benchmark the number of steps per second
        of an environment of type JaxEnv.

        Args:
            environment: JaxEnv to benchmark on.
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
        self._n_steps = n_steps
        self._batch_size = batch_size
        self._rng = hk.PRNGSequence(seed)
        self.step_fn: Callable[
            [State, Action], Tuple[State, TimeStep]
        ] = environment.step
        self.reset_fn: Callable[[PRNGKey], Tuple[State, TimeStep]] = environment.reset
        self.run_steps_jitted = jit(vmap(self._run_steps))
        if not isinstance(environment.action_spec(), specs.BoundedArray):
            action_spec = environment.action_spec()
            raise TypeError(
                f"action spec must be of type BoundedArray, got "
                f"{action_spec} of type {type(action_spec)}."
            )

    def _acting(self, key: PRNGKey) -> Action:
        """Select a random action for a specs.BoundedArray action space.

        Args:
            key: random key to sample an action.

        Returns:
            action randomly selected in the discrete interval given by
                specs.BoundedArray.

        """
        action = random.randint(
            key,
            self._environment.action_spec().shape,
            0,
            self._environment.action_spec()._num_values,
        )
        return action

    @partial(jit, static_argnums=0)
    def _learning(self, params: ArrayTree, data: Any) -> ArrayTree:
        """Computes a fake loss function.

        Args:
            params: parameters of the agent.
            data: batch of data used for training.

        Returns:
            loss

        """
        loss = jnp.float32(0.0)
        return loss

    def _run_steps(self, carry: Dict) -> Tuple[Dict, TimeStep]:
        """Runs a sequence of self._n_steps steps in the environment.

        Args:
            carry: dictionary containing:
                - episode_count: current number of episodes finished during
                    the sequence.
                - key: random key used to reset or take an action.
                - reset: boolean informing whether to reset the state or not.
                - state: current State of the environment.

        Returns:
            carry: last carry after taking n steps in the environment.
            timesteps: sequence of timesteps observed while taking steps.

        """

        def _run_one_step_reset(_carry: Dict) -> Tuple[Dict, TimeStep]:
            """Resets the environment because _carry['reset'] is True."""
            next_key, reset_key = random.split(_carry["key"])
            next_state, next_timestep = self.reset_fn(reset_key)  # type:ignore
            next_carry = {
                "episode_count": _carry["episode_count"] + 1,
                "key": next_key,
                "reset": False,
                "state": next_state,
            }
            return next_carry, next_timestep

        def _run_one_step_action(_carry: Dict) -> Tuple[Dict, TimeStep]:
            """Steps the environment because _carry['reset'] is False."""
            next_key, action_key = random.split(_carry["key"])
            action = self._acting(action_key)
            next_state, next_timestep = self.step_fn(_carry["state"], action)
            next_carry = {
                "episode_count": _carry["episode_count"],
                "key": next_key,
                "reset": next_timestep.last(),
                "state": next_state,
            }
            return next_carry, next_timestep

        def _run_one_step(_carry: Dict) -> Tuple[Dict, TimeStep]:
            """Runs one step in the environment. The behavior is a function
            of _carry['reset'].

            Args:
                _carry: current carry.

            Returns:
                next_carry: updated carry after running one step.
                next_timestep: timestep observed after the step.

            """
            next_carry, next_timestep = lax.cond(
                _carry["reset"],
                _run_one_step_reset,
                _run_one_step_action,
                _carry,
            )
            return next_carry, next_timestep

        carry, timesteps = lax.scan(
            lambda _carry, _: _run_one_step(_carry),
            carry,
            xs=None,
            length=self._n_steps,
            unroll=1,
        )
        return carry, timesteps

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

        episode_count, step_count = 0, 0
        params = None
        with TimeIt(tag="Compilation", print_=print_):
            keys = random.split(next(self._rng), self._batch_size)
            states, timesteps = vmap(self.reset_fn)(keys)
            carries = {
                "episode_count": jnp.zeros((self._batch_size,), int),
                "key": random.split(next(self._rng), self._batch_size),
                "reset": jnp.zeros((self._batch_size,), bool),
                "state": states,
            }
            self.run_steps_jitted(carries)

        while not should_terminate(episode_count, step_count, num_episodes, num_steps):
            with TimeIt(
                tag="Run", frames=self._n_steps * self._batch_size, ms=ms, print_=print_
            ):
                carries, ys = self.run_steps_jitted(carries)
                _ = self._learning(params, ys)
            episode_count += carries["episode_count"].sum()
            step_count += self._n_steps * self._batch_size
