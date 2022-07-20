import chex
import jax
import jax.numpy as jnp
import pytest
from jax import random

from jumanji import specs
from jumanji.binpack.env import BinPack
from jumanji.binpack.specs import EMSSpec, ItemSpec, ObservationSpec
from jumanji.binpack.types import State
from jumanji.testing.fakes import FakeAgent
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import TimeStep
from validation import JaxEnvironmentLoop


def test_binpack__reset(binpack_env: BinPack) -> None:
    """Validates the jitted reset of the environment."""
    chex.clear_trace_counter()
    reset_fn = jax.jit(chex.assert_max_traces(binpack_env.reset, n=1))

    key = random.PRNGKey(0)
    state, timestep, _ = reset_fn(key)
    # Call again to check it does not compile twice.
    state, timestep, _ = reset_fn(key)
    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # reset function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(state)
    assert state.ems_mask.any()
    assert state.items_mask.any()
    assert jnp.any(state.action_mask)


def test_binpack__spec(binpack_env: BinPack) -> None:
    """Validates the observation and action spec of the BinPack environment."""
    observation_spec = binpack_env.observation_spec()
    assert isinstance(observation_spec, ObservationSpec)
    assert isinstance(observation_spec.ems_spec, EMSSpec)
    assert isinstance(observation_spec.ems_mask_spec, specs.BoundedArray)
    assert isinstance(observation_spec.items_spec, ItemSpec)
    assert isinstance(observation_spec.items_mask_spec, specs.BoundedArray)
    assert isinstance(observation_spec.items_placed_spec, specs.BoundedArray)
    assert isinstance(observation_spec.action_mask_spec, specs.BoundedArray)
    assert isinstance(binpack_env.action_spec(), specs.BoundedArray)


def test_binpack__step(binpack_env: BinPack) -> None:
    """Validates the jitted step of the environment."""
    # TODO: Do more checks when step function implemented.
    chex.clear_trace_counter()
    step_fn = jax.jit(chex.assert_max_traces(binpack_env.step, n=1))

    key = random.PRNGKey(0)
    state, timestep, _ = binpack_env.reset(key)

    action = binpack_env.action_spec().generate_value()
    state, timestep, _ = step_fn(state, action)
    # Call again to check it does not compile twice.
    state, timestep, _ = step_fn(state, action)


def test_binpack__does_not_smoke(
    binpack_env: BinPack, capsys: pytest.CaptureFixture
) -> None:
    """Tests that we can run the jitted JaxEnvironmentLoop without any errors."""
    fake_agent = FakeAgent(binpack_env.action_spec())
    jax_environment_loop = JaxEnvironmentLoop(
        binpack_env, fake_agent, n_steps=1, batch_size=2
    )
    jax_environment_loop.run(num_steps=3)
    assert capsys.readouterr().out
