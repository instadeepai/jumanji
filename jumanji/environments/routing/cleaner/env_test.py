# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import chex
import jax
import jax.numpy as jnp
import pytest

from jumanji.environments.routing.cleaner.constants import CLEAN, DIRTY, WALL
from jumanji.environments.routing.cleaner.env import Cleaner
from jumanji.environments.routing.cleaner.generator import Generator
from jumanji.environments.routing.cleaner.types import Observation, State
from jumanji.testing.env_not_smoke import check_env_does_not_smoke
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import StepType, TimeStep

SAMPLE_GRID = jnp.array(
    [
        [CLEAN, DIRTY, WALL, DIRTY, DIRTY],
        [WALL, DIRTY, WALL, DIRTY, WALL],
        [DIRTY, DIRTY, DIRTY, DIRTY, WALL],
        [DIRTY, WALL, WALL, DIRTY, WALL],
        [DIRTY, WALL, DIRTY, DIRTY, DIRTY],
    ],
    dtype=jnp.int8,
)


class DummyGenerator(Generator):
    """Dummy generator, generate an instance of size 5x5 with 3 agents."""

    def __init__(self) -> None:
        super(DummyGenerator, self).__init__(num_rows=5, num_cols=5, num_agents=3)

    def __call__(self, key: chex.PRNGKey) -> State:
        agents_locations = jnp.zeros((self.num_agents, 2), int)
        return State(
            grid=SAMPLE_GRID,
            agents_locations=agents_locations,
            action_mask=None,
            step_count=jnp.array(0, jnp.int32),
            key=key,
        )


class TestCleaner:
    @pytest.fixture
    def cleaner(self) -> Cleaner:
        generator = DummyGenerator()
        return Cleaner(generator=generator)

    @pytest.fixture
    def key(self) -> chex.PRNGKey:
        return jax.random.PRNGKey(0)

    def test_cleaner__reset_jit(self, cleaner: Cleaner) -> None:
        """Confirm that the reset is only compiled once when jitted."""
        chex.clear_trace_counter()
        reset_fn = jax.jit(chex.assert_max_traces(cleaner.reset, n=1))
        key = jax.random.PRNGKey(0)
        state, timestep = reset_fn(key)

        # Call again to check it does not compile twice
        state, timestep = reset_fn(key)
        assert isinstance(timestep, TimeStep)
        assert isinstance(state, State)

    def test_cleaner__reset(self, cleaner: Cleaner, key: chex.PRNGKey) -> None:
        reset_fn = jax.jit(cleaner.reset)
        state, timestep = reset_fn(key)

        assert isinstance(timestep, TimeStep)
        assert isinstance(state, State)

        assert jnp.all(state.agents_locations == jnp.zeros((cleaner.num_agents, 2)))
        assert jnp.sum(state.grid == CLEAN) == 1  # Only the top-left tile is clean
        assert state.step_count == 0

        assert_is_jax_array_tree(state)

    def test_cleaner__step_jit(self, cleaner: Cleaner) -> None:
        """Confirm that the step is only compiled once when jitted."""
        key = jax.random.PRNGKey(0)
        state, timestep = cleaner.reset(key)
        action = jnp.array([1, 2, 3], jnp.int32)

        chex.clear_trace_counter()
        step_fn = jax.jit(chex.assert_max_traces(cleaner.step, n=1))
        next_state, next_timestep = step_fn(state, action)

        # Call again to check it does not compile twice
        next_state, next_timestep = step_fn(state, action)
        assert isinstance(next_timestep, TimeStep)
        assert isinstance(next_state, State)

    def test_cleaner__step(self, cleaner: Cleaner, key: chex.PRNGKey) -> None:
        initial_state, timestep = cleaner.reset(key)

        step_fn = jax.jit(cleaner.step)

        # First action: all agents move right
        actions = jnp.array([1] * cleaner.num_agents)
        state, timestep = step_fn(initial_state, actions)
        # Assert only one tile changed, on the right of the initial pos
        assert jnp.sum(state.grid != initial_state.grid) == 1
        assert state.grid[0, 1] == CLEAN
        assert timestep.reward == 1 - cleaner.penalty_per_timestep
        assert jnp.all(state.agents_locations == jnp.array([0, 1]))

        # Second action: agent 0 and 2 move down, agent 1 moves left
        actions = jnp.array([2, 3, 2])
        state, timestep = step_fn(state, actions)
        # Assert only two tiles changed in total since the reset
        assert jnp.sum(state.grid != initial_state.grid) == 2
        assert state.grid[0, 1] == CLEAN
        assert state.grid[1, 1] == CLEAN
        assert timestep.reward == 1 - cleaner.penalty_per_timestep
        assert timestep.step_type == StepType.MID

        assert jnp.all(state.agents_locations[0] == jnp.array([1, 1]))
        assert jnp.all(state.agents_locations[1] == jnp.array([0, 0]))
        assert jnp.all(state.agents_locations[2] == jnp.array([1, 1]))

    def test_cleaner__step_invalid_action(
        self, cleaner: Cleaner, key: chex.PRNGKey
    ) -> None:
        state, _ = cleaner.reset(key)

        step_fn = jax.jit(cleaner.step)
        # Invalid action for agent 0, valid for 1 and 2
        actions = jnp.array([0, 1, 1])
        state, timestep = step_fn(state, actions)

        assert timestep.step_type == StepType.LAST

        assert jnp.all(state.agents_locations[0] == jnp.array([0, 0]))
        assert jnp.all(state.agents_locations[1] == jnp.array([0, 1]))
        assert jnp.all(state.agents_locations[2] == jnp.array([0, 1]))

        assert timestep.reward == 1 - cleaner.penalty_per_timestep

    def test_cleaner__initial_action_mask(
        self, cleaner: Cleaner, key: chex.PRNGKey
    ) -> None:
        state, _ = cleaner.reset(key)

        # All agents can only move right in the initial state
        expected_action_mask = jnp.array(
            [[False, True, False, False] for _ in range(cleaner.num_agents)]
        )

        assert jnp.all(state.action_mask == expected_action_mask)

        action_mask = cleaner._compute_action_mask(state.grid, state.agents_locations)
        assert jnp.all(action_mask == expected_action_mask)

    def test_cleaner__action_mask(self, cleaner: Cleaner, key: chex.PRNGKey) -> None:
        state, _ = cleaner.reset(key)

        # Test action mask for different agent locations
        agents_locations = jnp.array([[1, 1], [2, 2], [4, 4]])
        action_mask = cleaner._compute_action_mask(state.grid, agents_locations)

        assert jnp.all(action_mask[0] == jnp.array([True, False, True, False]))
        assert jnp.all(action_mask[1] == jnp.array([False, True, False, True]))
        assert jnp.all(action_mask[2] == jnp.array([False, False, False, True]))

    def test_cleaner__does_not_smoke(self, cleaner: Cleaner) -> None:
        def select_actions(key: chex.PRNGKey, observation: Observation) -> chex.Array:
            @jax.vmap  # map over the keys and agents
            def select_action(
                key: chex.PRNGKey, agent_action_mask: chex.Array
            ) -> chex.Array:
                return jax.random.choice(
                    key, jnp.arange(4), p=agent_action_mask.flatten()
                )

            subkeys = jax.random.split(key, cleaner.num_agents)
            return select_action(subkeys, observation.action_mask)

        check_env_does_not_smoke(cleaner, select_actions)

    def test_cleaner__compute_extras(self, cleaner: Cleaner, key: chex.PRNGKey) -> None:
        state, _ = cleaner.reset(key)

        extras = cleaner._compute_extras(state)
        assert list(extras.keys()) == ["ratio_dirty_tiles", "num_dirty_tiles"]
        assert 0 <= extras["ratio_dirty_tiles"] <= 1
        grid = state.grid
        assert extras["ratio_dirty_tiles"] == jnp.sum(grid == DIRTY) / jnp.sum(
            grid != WALL
        )
        assert extras["num_dirty_tiles"] == jnp.sum(grid == DIRTY)
