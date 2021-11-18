import dm_env
import jax.numpy as jnp
import pytest
from jax import lax

from jumanji.jax.types import TimeStep, restart, termination, transition, truncation


def test_dm_env_timestep__not_jaxtype() -> None:
    """Validates that dm_env.TimeStep is not a valid Jax type, and hence cannot
    be used in lax.select or lax.cond. Indeed, when not using a Jax type, both lax.select and
    lax.cond fail. This test checks the behavior of neither termination nor transition, but rather
    makes sure that dm_env.TimeStep is not a Jax type. If it is, the latter can be used in
    replacement of the custom implementation of TimeStep.
    """

    def _get_termination_transition() -> dm_env.TimeStep:
        """Returns either a termination or transition TimeStep."""
        return lax.select(
            True,
            dm_env.termination(reward=None, observation=None),
            dm_env.transition(reward=None, observation=None),
        )

    try:
        _ = _get_termination_transition()
        raise EnvironmentError(
            "dm_env.TimeStep now seems to be a Jax type, meaning it can replace "
            "custom implementation of TimeStep."
        )
    except Exception as ex:
        assert isinstance(ex, TypeError)


@pytest.mark.parametrize("done", (True, False))
def test_timestep__jaxtype(done: bool) -> None:
    """Validates that custom TimeStep from jumanji.jax.types is a valid Jax type.
    Checks that custom termination, transition, restart and truncation functions return
    a valid Jax type. This test checks the behavior of neither termination, transition,
    truncation nor restart, but rather makes sure that TimeStep is a Jax type.
    """

    def _get_termination_transition() -> TimeStep:
        """Returns either a termination or transition TimeStep."""
        return lax.cond(  # type: ignore
            done,
            lambda _: termination(
                reward=jnp.zeros((), float), observation=jnp.zeros((), float)
            ),
            lambda _: transition(
                reward=jnp.zeros((), float), observation=jnp.zeros((), float)
            ),
            None,
        )

    timestep = _get_termination_transition()
    assert isinstance(timestep, TimeStep)
    assert not isinstance(timestep, dm_env.TimeStep)

    def _get_restart_truncation() -> TimeStep:
        """Returns either a restart or truncation TimeStep."""
        return lax.cond(  # type: ignore
            done,
            lambda _: restart(observation=jnp.zeros((), float)),
            lambda _: truncation(
                reward=jnp.zeros((), float), observation=jnp.zeros((), float)
            ),
            None,
        )

    timestep = _get_restart_truncation()
    assert isinstance(timestep, TimeStep)
    assert not isinstance(timestep, dm_env.TimeStep)


def test_timestep__restart() -> None:
    """Validates that restart function returns the desired TimeStep."""
    observation = jnp.ones(5, float)
    timestep = restart(observation)
    assert jnp.all(timestep.observation == observation)
    assert timestep.step_type is dm_env.StepType.FIRST
    assert timestep.reward == 0.0
    assert timestep.discount == 1.0


def test_timestep__transition() -> None:
    """Validates that transition function returns the desired TimeStep."""
    observation = jnp.ones(5, float)
    reward = jnp.array(2.0)
    timestep = transition(reward, observation)
    assert jnp.all(timestep.observation == observation)
    assert timestep.step_type is dm_env.StepType.MID
    assert timestep.reward == reward
    assert timestep.discount == 1.0


def test_timestep__truncation() -> None:
    """Validates that truncation function returns the desired TimeStep."""
    observation = jnp.ones(5, float)
    reward = jnp.array(2.0)
    discount = jnp.array(0.8)
    timestep = truncation(reward, observation, discount)
    assert jnp.all(timestep.observation == observation)
    assert timestep.step_type is dm_env.StepType.LAST
    assert timestep.reward == reward
    assert timestep.discount == discount


def test_timestep__termination() -> None:
    """Validates that termination function returns the desired TimeStep."""
    observation = jnp.ones(5, float)
    reward = jnp.array(2.0)
    timestep = termination(reward, observation)
    assert jnp.all(timestep.observation == observation)
    assert timestep.step_type is dm_env.StepType.LAST
    assert timestep.reward == reward
    assert timestep.discount == 0.0
