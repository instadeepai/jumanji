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

from jumanji.environments.routing.maze.env import Maze
from jumanji.environments.routing.maze.generator import RandomGenerator, ToyGenerator
from jumanji.environments.routing.maze.types import Position, State
from jumanji.testing.env_not_smoke import (
    check_env_does_not_smoke,
    check_env_specs_does_not_smoke,
)
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import StepType, TimeStep


class TestMazeEnvironment:
    @pytest.fixture(scope="module")
    def maze(self) -> Maze:
        """Instantiates a default Maze environment."""
        generator = RandomGenerator(num_rows=5, num_cols=5)
        return Maze(generator=generator, time_limit=15)

    def test_maze__reset(self, maze: Maze) -> None:
        reset_fn = jax.jit(maze.reset)
        key = jax.random.PRNGKey(0)
        state, timestep = reset_fn(key)

        assert isinstance(timestep, TimeStep)
        assert isinstance(state, State)
        assert state.step_count == 0
        assert_is_jax_array_tree(state)

        # Check that the agent and target positions are not the same
        assert state.agent_position != state.target_position

        # Check that the agent and target are in a non-wall cell
        assert not state.walls[tuple(state.agent_position)]
        assert not state.walls[tuple(state.target_position)]

    def test_env__reset_jit(self, maze: Maze) -> None:
        """Confirm that the reset is only compiled once when jitted."""
        chex.clear_trace_counter()
        reset_fn = jax.jit(chex.assert_max_traces(maze.reset, n=1))
        key = jax.random.PRNGKey(0)
        state, timestep = reset_fn(key)

        # Call again to check it does not compile twice
        state, timestep = reset_fn(key)
        assert isinstance(timestep, TimeStep)
        assert isinstance(state, State)

    def test_env__step_jit(self, maze: Maze) -> None:
        """Confirm that the step is only compiled once when jitted."""
        key = jax.random.PRNGKey(0)
        state, timestep = maze.reset(key)
        assert isinstance(timestep, TimeStep)
        assert isinstance(state, State)
        action = jnp.array(2, jnp.int32)

        chex.clear_trace_counter()
        step_fn = jax.jit(chex.assert_max_traces(maze.step, n=1))
        next_state, next_timestep = step_fn(state, action)

        # Call again to check it does not compile twice
        next_state, next_timestep = step_fn(state, action)
        assert isinstance(next_timestep, TimeStep)
        assert isinstance(next_state, State)

    def test_maze__random_agent_start(self, maze: Maze) -> None:
        key1, key2 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
        state1, _ = maze.reset(key1)
        state2, _ = maze.reset(key2)

        # Check random positions are different
        assert state1.agent_position != state2.agent_position
        assert state1.target_position != state2.target_position

    def test_maze__step(self, maze: Maze) -> None:
        key = jax.random.PRNGKey(0)
        state, _ = maze.reset(key)

        # Fixed agent start state
        agent_position = Position(row=4, col=0)
        initial_state = State(
            agent_position=agent_position,
            target_position=state.target_position,
            walls=state.walls,
            action_mask=maze._compute_action_mask(state.walls, agent_position),
            key=state.key,
            step_count=jnp.array(0, jnp.int32),
        )

        step_fn = jax.jit(maze.step)

        # Agent takes a step right
        action = jnp.array(1, jnp.int32)
        state, timestep = step_fn(initial_state, action)

        assert timestep.reward == 0
        assert timestep.step_type == StepType.MID
        assert state.agent_position == Position(row=4, col=1)

        # Agent takes a step right
        action = jnp.array(1, jnp.int32)
        state, timestep = step_fn(state, action)

        assert timestep.reward == 0
        assert timestep.step_type == StepType.MID
        assert state.agent_position == Position(row=4, col=2)

        # Agent takes a step up
        action = jnp.array(0, jnp.int32)
        state, timestep = step_fn(state, action)

        assert timestep.reward == 0
        assert timestep.step_type == StepType.MID
        assert state.agent_position == Position(row=3, col=2)

        # Agent takes a step up
        action = jnp.array(0, jnp.int32)
        state, timestep = step_fn(state, action)

        assert timestep.reward == 0
        assert timestep.step_type == StepType.MID
        assert state.agent_position == Position(row=2, col=2)

        # Agent fails to take a step left due to wall
        action = jnp.array(3, jnp.int32)
        state, timestep = step_fn(state, action)

        assert timestep.reward == 0
        assert timestep.step_type == StepType.MID
        assert state.agent_position == Position(row=2, col=2)

    def test_maze__action_mask(self, maze: Maze) -> None:
        key = jax.random.PRNGKey(0)
        state, _ = maze.reset(key)

        # Fixed agent start state
        agent_position = Position(row=4, col=0)

        # The agent can only move up or right in the initial state
        expected_action_mask = jnp.array([True, True, False, False])
        action_mask = maze._compute_action_mask(state.walls, agent_position)
        assert jnp.all(action_mask == expected_action_mask)

        # Check another position
        another_position = Position(row=2, col=2)

        # The agent can move up, right or down
        expected_action_mask = jnp.array([True, True, True, False])
        action_mask = maze._compute_action_mask(state.walls, another_position)
        assert jnp.all(action_mask == expected_action_mask)

    def test_maze__reward(self, maze: Maze) -> None:
        key = jax.random.PRNGKey(0)
        state, timestep = maze.reset(key)

        # Fixed agent and target positions
        agent_position = Position(row=4, col=0)
        target_position = Position(row=0, col=2)

        state = State(
            agent_position=agent_position,
            target_position=target_position,
            walls=state.walls,
            action_mask=maze._compute_action_mask(state.walls, agent_position),
            key=state.key,
            step_count=jnp.array(0, jnp.int32),
        )

        actions = [1, 1, 0, 0, 0, 0]

        for a in actions:
            assert timestep.reward == 0
            assert timestep.step_type < StepType.LAST
            state, timestep = maze.step(state, a)

        # Final step into the target
        assert timestep.reward == 1
        assert timestep.last()
        assert state.agent_position == state.target_position

    def test_maze__toy_generator(self) -> None:
        key = jax.random.PRNGKey(0)

        toy_generator = ToyGenerator()
        maze = Maze(generator=toy_generator, time_limit=25)
        state, timestep = maze.reset(key)

        # Fixed agent and target positions
        agent_position = Position(row=4, col=0)
        target_position = Position(row=0, col=3)

        state = State(
            agent_position=agent_position,
            target_position=target_position,
            walls=state.walls,
            action_mask=maze._compute_action_mask(state.walls, agent_position),
            key=state.key,
            step_count=jnp.array(0, jnp.int32),
        )

        actions = [1, 1, 0, 0, 0, 0, 1]

        for a in actions:
            assert timestep.reward == 0
            assert timestep.step_type < StepType.LAST
            state, timestep = maze.step(state, a)

        # Final step into the target
        assert timestep.reward == 1
        assert timestep.last()
        assert state.agent_position == state.target_position

    def test_maze__does_not_smoke(self, maze: Maze) -> None:
        check_env_does_not_smoke(maze)

    def test_maze__specs_does_not_smoke(self, maze: Maze) -> None:
        """Test that we can access specs without any errors."""
        check_env_specs_does_not_smoke(maze)
