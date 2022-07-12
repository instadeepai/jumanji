import chex
import jax
import pytest
from jax import numpy as jnp
from jax import random

from jumanji.jax.tsp.env import TSP
from jumanji.jax.tsp.types import State
from jumanji.jax.types import StepType, TimeStep
from jumanji.testing.fakes import FakeAgent
from jumanji.testing.pytrees import assert_is_jax_array_tree
from validation import JaxEnvironmentLoop


@pytest.fixture
def tsp_env() -> TSP:
    """Instantiates a default TSP environment."""
    return TSP()


def test_tsp__reset(tsp_env: TSP) -> None:
    """Validates the jitted reset of the environment."""
    reset_fn = jax.jit(tsp_env.reset)
    key = random.PRNGKey(0)
    state, timestep, _ = reset_fn(key)
    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)
    assert state.visited_mask[state.position] == 1
    assert state.visited_mask.sum() == 1
    assert state.order[0] == state.position
    assert state.num_visited == 1

    assert_is_jax_array_tree(state)


def test_tsp__step(tsp_env: TSP) -> None:
    """Validates the jitted step of the environment."""
    chex.clear_trace_counter()

    step_fn = chex.assert_max_traces(tsp_env.step, n=1)
    step_fn = jax.jit(step_fn)

    key = random.PRNGKey(0)
    state, timestep, _ = tsp_env.reset(key)

    last_action = state.position
    new_action = last_action - 1 if last_action > 0 else 0

    new_state, next_timestep, _ = step_fn(state, new_action)

    # Check that the state has changed
    assert not jnp.array_equal(new_state.position, state.position)
    assert not jnp.array_equal(new_state.visited_mask, state.visited_mask)
    assert not jnp.array_equal(new_state.order, state.order)
    assert not jnp.array_equal(new_state.num_visited, state.num_visited)

    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # step function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(new_state)

    # Check token was inserted as expected
    assert new_state.visited_mask[new_action] == 1
    assert new_state.visited_mask.sum() == 2

    # New step with same action should be invalid
    state = new_state

    new_state, next_timestep, _ = step_fn(state, new_action)

    # Check that the state has not changed
    assert jnp.array_equal(new_state.position, state.position)
    assert jnp.array_equal(new_state.visited_mask, state.visited_mask)
    assert jnp.array_equal(new_state.order, state.order)
    assert jnp.array_equal(new_state.num_visited, state.num_visited)


def test_tsp__does_not_smoke(tsp_env: TSP, capsys: pytest.CaptureFixture) -> None:
    """Tests that we can run the jitted JaxEnvironmentLoop without any errors."""
    fake_agent = FakeAgent(tsp_env.action_spec())
    jax_environment_loop = JaxEnvironmentLoop(
        tsp_env, fake_agent, n_steps=1, batch_size=2
    )
    jax_environment_loop.run(num_steps=3)
    assert capsys.readouterr().out


def test_tsp__trajectory_action(tsp_env: TSP) -> None:
    """
    Checks that the agent stops when there are no more cities to be selected and that the appropriate reward is
    received. The testing loop ensures that no city is selected twice.
    """
    key = random.PRNGKey(0)
    state, timestep, _ = tsp_env.reset(key)

    while not timestep.last():
        # Check that there are cities that have not been selected yet.
        assert state.num_visited < tsp_env.problem_size
        assert state.visited_mask.sum() < tsp_env.problem_size

        # Check that the reward is 0 while trajectory is not done.
        assert timestep.reward == 0

        state, timestep, _ = tsp_env.step(
            state, (state.position + 1) % tsp_env.problem_size
        )

    # Check that the reward is negative when trajectory is done.
    assert timestep.reward < 0

    # Check that no action can be taken (all cities have been selected)
    assert state.num_visited == tsp_env.problem_size
    assert state.visited_mask.sum() == tsp_env.problem_size

    assert timestep.last()


def test_tsp__invalid_action(tsp_env: TSP) -> None:
    """Checks that an invalid action leads to a termination and the appropriate reward is received."""
    key = random.PRNGKey(0)
    state, timestep, _ = tsp_env.reset(key)

    first_position = state.position
    actions = (
        jnp.array([first_position + 1, first_position + 2, first_position + 2])
        % tsp_env.problem_size
    )

    for a in actions:
        assert timestep.reward == 0
        assert timestep.step_type < StepType.LAST
        state, timestep, _ = tsp_env.step(state, a)

    # Last action is invalid because it was already taken
    assert timestep.reward < 0
    assert timestep.last()
