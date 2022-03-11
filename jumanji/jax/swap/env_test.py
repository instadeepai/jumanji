import chex
import jax
import jax.numpy as jnp
import pytest
from jax import random

from jumanji.jax.swap import Swap
from jumanji.jax.swap.types import State
from jumanji.jax.types import TimeStep
from jumanji.jax.wrappers import DeepMindEnvWrapper
from jumanji.testing.fakes import FakeAgent
from jumanji.testing.pytrees import assert_is_jax_array_tree
from validation.benchmark_loops import DeepMindEnvBenchmarkLoop
from validation.environment_loops import JaxEnvironmentLoop


@pytest.fixture
def swap_env() -> Swap:
    """Instantiates a default Swap environment."""
    return Swap()


@pytest.mark.parametrize("swap_env", [()], indirect=True)
def test_swap__reset(swap_env: Swap) -> None:
    """Validates the jitted reset of the environment."""
    reset_fn = jax.jit(swap_env.reset)
    key1, key2 = random.PRNGKey(0), random.PRNGKey(1)
    state1, timestep1, _ = reset_fn(key1)
    state2, timestep2, _ = reset_fn(key2)
    assert isinstance(timestep1, TimeStep)
    assert isinstance(state1, State)
    assert state1.step_count == 0
    # Check random initialization
    assert jnp.any(state1.agent_pos != state2.agent_pos)
    assert jnp.any(state1.blue_pos != state2.blue_pos)
    assert jnp.any(state1.red_pos != state2.red_pos)
    assert not jnp.any(state1.key == state2.key)
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # reset function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(state1)


@pytest.mark.parametrize("swap_env", [()], indirect=True)
def test_swap__step(swap_env: Swap) -> None:
    """Validates the jitted step function of the environment."""
    step_fn = jax.jit(swap_env.step)
    state_key, action_key = random.split(random.PRNGKey(0))
    state, timestep, _ = swap_env.reset(state_key)
    # Sample two different actions
    action1, action2 = random.choice(
        action_key,
        jnp.arange(swap_env.action_spec()._num_values),
        shape=(2,),
        replace=False,
    )
    new_state1, timestep1, _ = step_fn(state, action1)
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # step function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(new_state1)
    # Check that the state has changed
    assert new_state1.step_count == state.step_count + 1
    assert jnp.any(new_state1.agent_pos != state.agent_pos)
    # Check that two different actions lead to two different states
    new_state2, timestep2, _ = step_fn(state, action2)
    assert jnp.any(new_state1.agent_pos != new_state2.agent_pos)
    # Check that the state update and timestep creation work as expected
    agent_pos = state.agent_pos
    moves = {
        0: jnp.array((agent_pos[0] - 1, agent_pos[1])),  # Up
        1: jnp.array((agent_pos[0], agent_pos[1] + 1)),  # Right
        2: jnp.array((agent_pos[0] + 1, agent_pos[1])),  # Down
        3: jnp.array((agent_pos[0], agent_pos[1] - 1)),  # Left
    }
    for action, new_agent_pos in moves.items():
        new_state, timestep, _ = step_fn(state, action)
        assert jnp.array_equal(new_state.agent_pos, new_agent_pos)
        assert jnp.array_equal(
            timestep.observation[0],
            jax.nn.one_hot(new_agent_pos, max(swap_env.n_rows, swap_env.n_cols)),
        )


@pytest.mark.parametrize("swap_env", [()], indirect=True)
def test_swap__target_reached(swap_env: Swap) -> None:
    """Validates the effect of reaching a target."""
    step_fn = jax.jit(swap_env.step)
    state, *_ = swap_env.reset(random.PRNGKey(0))
    state.agent_pos = jnp.array((2, 2), int)
    state.blue_pos = jnp.array((2, 3), int)
    state.red_pos = jnp.array((3, 2), int)
    new_state_blue, timestep_blue, _ = step_fn(state, 1)  # blue target is on the right
    new_state_red, timestep_red, _ = step_fn(state, 2)  # red target is below
    # Reaching blue target
    assert jnp.array_equal(new_state_blue.agent_pos, state.blue_pos)
    assert timestep_blue.reward == 1
    assert not jnp.array_equal(new_state_blue.blue_pos, state.blue_pos)
    assert jnp.array_equal(new_state_blue.red_pos, state.red_pos)
    # Reaching red target
    assert jnp.array_equal(new_state_red.agent_pos, state.red_pos)
    assert timestep_red.reward == -1
    assert not jnp.array_equal(new_state_red.red_pos, state.red_pos)
    assert jnp.array_equal(new_state_red.blue_pos, state.blue_pos)

    # After 100_000 steps, the targets are flipped.
    swap_env.swap_period = 100_000
    state.step_count = 100_000
    new_state_blue, timestep_blue, _ = step_fn(state, 1)  # blue target is on the right
    new_state_red, timestep_red, _ = step_fn(state, 2)  # red target is below
    # Reaching blue target
    assert jnp.array_equal(new_state_blue.agent_pos, state.blue_pos)
    assert timestep_blue.reward == -1
    assert not jnp.array_equal(new_state_blue.blue_pos, state.blue_pos)
    assert jnp.array_equal(new_state_blue.red_pos, state.red_pos)
    # Reaching red target
    assert jnp.array_equal(new_state_red.agent_pos, state.red_pos)
    assert timestep_red.reward == 1
    assert not jnp.array_equal(new_state_red.red_pos, state.red_pos)
    assert jnp.array_equal(new_state_red.blue_pos, state.blue_pos)


@pytest.mark.parametrize("swap_env", [()], indirect=True)
def test_swap__episode_terminates(swap_env: Swap) -> None:
    """Validates that the episode terminates after swapping targets twice."""
    swap_env.swap_period = 5
    step_fn = jax.jit(swap_env.step)
    state, timestep, _ = swap_env.reset(random.PRNGKey(0))
    while not timestep.last():
        state, timestep, _ = step_fn(state, 0)


def test_swap__agent_remains_on_the_board() -> None:
    """Validates that the agent stays on the board whatever action it takes."""
    n_rows, n_cols = 3, 5
    swap_env = Swap(n_rows=n_rows, n_cols=n_cols)
    step_fn = jax.jit(swap_env.step)
    for action in range(4):
        state, timestep, _ = swap_env.reset(random.PRNGKey(0))
        for _ in range(6):
            state, timestep, _ = step_fn(state, action)
        assert jnp.all(state.agent_pos >= jnp.array((0, 0), int)) and jnp.all(
            state.agent_pos <= jnp.array((n_rows, n_cols), int)
        )


def test_swap__item_sampling() -> None:
    """Validates that both items are sampled correctly."""
    swap_env = Swap(n_rows=2, n_cols=2)
    reset_key, sample_key = random.split(random.PRNGKey(0))
    state, timestep, _ = swap_env.reset(reset_key)

    @jax.jit
    def sample_is_correct(key: chex.PRNGKey) -> chex.Array:
        blue_key, red_key = random.split(key)
        new_red_pos = swap_env._sample_pos(
            agent_pos=state.agent_pos, item_pos=state.blue_pos, key=red_key
        )
        new_red_pos_is_good = jnp.logical_and(
            jnp.any(new_red_pos != state.blue_pos),
            jnp.any(new_red_pos != state.agent_pos),
        )
        new_blue_pos = swap_env._sample_pos(
            agent_pos=state.agent_pos, item_pos=state.red_pos, key=blue_key
        )
        new_blue_pos_is_good = jnp.logical_and(
            jnp.any(new_blue_pos != state.red_pos),
            jnp.any(new_blue_pos != state.agent_pos),
        )
        return jnp.logical_and(new_red_pos_is_good, new_blue_pos_is_good)

    keys = random.split(sample_key, 10)
    assert jax.vmap(sample_is_correct)(keys).all()


@pytest.mark.parametrize("swap_env", [()], indirect=True)
def test_swap__does_not_smoke(swap_env: Swap, capsys: pytest.CaptureFixture) -> None:
    """Test that we can run the jitted JaxEnvironmentLoop without any errors."""
    fake_agent = FakeAgent()
    swap_env.swap_period = 5
    jax_environment_loop = JaxEnvironmentLoop(
        swap_env, fake_agent, n_steps=1, batch_size=2
    )
    jax_environment_loop.run(num_steps=3)
    deep_mind_env_benchmark_loop = DeepMindEnvBenchmarkLoop(
        DeepMindEnvWrapper(swap_env)
    )
    deep_mind_env_benchmark_loop.run(num_steps=3)
    assert capsys.readouterr().out
