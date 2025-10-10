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
from typing import Tuple

import chex
import jax
import pytest
from jax import numpy as jnp

from jumanji.environments.routing.connector.generator import (
    RandomWalkGenerator,
    UniformRandomGenerator,
)
from jumanji.environments.routing.connector.types import Agent
from jumanji.environments.routing.connector.utils import get_action_masks, get_position, get_target


@pytest.fixture
def uniform_random_generator() -> UniformRandomGenerator:
    """Creates a generator with grid size of 5 and 3 agents."""
    return UniformRandomGenerator(grid_size=5, num_agents=3)


def test_uniform_random_generator__call(
    uniform_random_generator: UniformRandomGenerator, key: chex.PRNGKey
) -> None:
    """Test that generator generates valid boards."""
    state = uniform_random_generator(key)

    assert state.grid.shape == (5, 5)
    assert state.agents.position.shape == state.agents.target.shape == (3, 2)

    # Check grid has head and target for each agent
    # and the starts and ends point to the correct heads and targets, respectively
    agents_on_grid = state.grid[jax.vmap(tuple)(state.agents.position)]
    targets_on_grid = state.grid[jax.vmap(tuple)(state.agents.target)]
    assert (agents_on_grid == jnp.array([get_position(i) for i in range(3)])).all()
    assert (targets_on_grid == jnp.array([get_target(i) for i in range(3)])).all()


def test_uniform_random_generator__no_retrace(
    uniform_random_generator: UniformRandomGenerator, key: chex.PRNGKey
) -> None:
    """Checks that generator only traces the function once and works when jitted."""
    keys = jax.random.split(key, 2)
    jitted_generator = jax.jit(chex.assert_max_traces((uniform_random_generator.__call__), n=1))

    for key in keys:
        jitted_generator(key)


### grids for testing
empty_grid = jnp.zeros((5, 5))
valid_starting_grid = jnp.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [2, 0, 0, 0, 5],
        [0, 0, 0, 0, 0],
        [0, 0, 8, 0, 0],
    ],
    dtype=jnp.int32,
)
valid_starting_grid_after_1_step = jnp.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 5, 4],
        [2, 0, 0, 0, 0],
        [0, 8, 7, 0, 0],
    ],
    dtype=jnp.int32,
)
valid_starting_grid_initialize_agents = jnp.array(
    [
        [0, 0, 0, 1, 0],
        [8, 0, 0, 2, 0],
        [7, 0, 0, 0, 0],
        [0, 0, 0, 5, 0],
        [0, 0, 0, 4, 0],
    ],
    dtype=jnp.int32,
)

valid_solved_grid_1 = jnp.array(
    [
        [1, 1, 1, 1, 2],
        [4, 4, 4, 4, 4],
        [7, 7, 7, 0, 5],
        [7, 0, 7, 7, 7],
        [7, 8, 0, 0, 7],
    ],
    dtype=jnp.int32,
)

valid_training_grid = jnp.array(
    [
        [0, 0, 0, 0, 0],
        [0, 8, 0, 0, 0],
        [0, 0, 6, 0, 9],
        [0, 0, 2, 0, 0],
        [3, 0, 0, 5, 0],
    ],
    dtype=jnp.int32,
)

valid_solved_grid_2 = jnp.array(
    [
        [0, 7, 7, 7, 7],
        [0, 8, 7, 7, 7],
        [1, 1, 6, 4, 9],
        [1, 1, 2, 4, 4],
        [3, 1, 1, 5, 4],
    ],
    dtype=jnp.int32,
)

grid_to_test_available_cells = jnp.array(
    [
        [1, 1, 1, 1, 2],
        [4, 4, 4, 4, 5],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=jnp.int32,
)
grids_after_1_agent_step = jnp.array(
    [
        [
            [0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [1, 0, 0, 0, 5],
            [0, 0, 0, 0, 0],
            [0, 0, 8, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [2, 0, 0, 0, 4],
            [0, 0, 0, 0, 0],
            [0, 0, 8, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [2, 0, 0, 0, 5],
            [0, 0, 8, 0, 0],
            [0, 0, 7, 0, 0],
        ],
    ],
)
### Agents for testing
agents_finished = Agent(
    id=jnp.array([0, 1, 2]),
    start=jnp.array([[2, 0], [2, 4], [4, 2]]),
    target=jnp.array([[-1, -1], [-1, -1], [-1, -1]]),
    position=jnp.array([[0, 4], [2, 4], [4, 1]]),
)
agents_reshaped_for_generator = Agent(
    id=jnp.arange(3),
    start=jnp.array([[0, 1, 4], [0, 0, 4]]),
    target=jnp.array([[-1, -1, -1], [-1, -1, -1]]),
    position=jnp.array([[0, 4, 4], [4, 0, 2]]),
)
agents_starting = Agent(
    id=jnp.array([0, 1, 2]),
    start=jnp.array([[2, 0], [2, 4], [4, 2]]),
    target=jnp.array([[-1, -1], [-1, -1], [-1, -1]]),
    position=jnp.array([[2, 0], [2, 4], [4, 2]]),
)

agents_starting_initialise_agents = Agent(
    id=jnp.array([0, 1, 2]),
    start=jnp.array([[0, 3], [4, 3], [2, 0]]),
    target=jnp.array([[-1, -1], [-1, -1], [-1, -1]]),
    position=jnp.array([[1, 3], [3, 3], [1, 0]]),
)
agents_starting_move_after_1_step = Agent(
    id=jnp.array([0, 1, 2]),
    start=jnp.array([[2, 0], [2, 4], [4, 2]]),
    target=jnp.array([[-1, -1], [-1, -1], [-1, -1]]),
    position=jnp.array([[3, 0], [2, 3], [4, 1]]),
)

agents_starting_move_1_step_up = Agent(
    id=jnp.array([0, 1, 2]),
    start=jnp.array([[2, 0], [2, 4], [4, 2]]),
    target=jnp.array([[-1, -1], [-1, -1], [-1, -1]]),
    position=jnp.array([[2, 1], [2, 3], [3, 2]]),
)

generate_board_agents = Agent(
    id=jnp.array([0, 1, 2]),
    start=jnp.array([[3, 2], [4, 3], [1, 1]]),
    target=jnp.array([[4, 0], [2, 2], [2, 4]]),
    position=jnp.array([[3, 2], [4, 3], [1, 1]]),
)

# Action masks for testing
action_mask_all_valid = jnp.array(
    [
        [True, True, True, True, True],
        [True, True, True, True, True],
        [True, True, True, True, True],
    ]
)
action_mask_none_valid = jnp.array(
    [
        [True, False, False, False, False],
        [True, False, False, False, False],
        [True, False, False, False, False],
    ]
)

key = jax.random.PRNGKey(0)
### keys for testing
key_1, key_2 = jax.random.split(key)


class TestRandomWalkGenerator:
    @pytest.fixture
    def random_walk_generator(self) -> RandomWalkGenerator:
        """Creates a generator with grid size of 5 and 3 agents."""
        return RandomWalkGenerator(grid_size=5, num_agents=3)

    def test_random_walk_generator__call(
        self,
        random_walk_generator: RandomWalkGenerator,
        key: chex.PRNGKey,
    ) -> None:
        """Tests that generator generates valid boards."""
        state = random_walk_generator(key)

        assert state.grid.shape == (5, 5)
        assert state.agents.position.shape == state.agents.target.shape == (3, 2)

        # Check grid has head and target for each agent
        # and the starts and ends point to the correct heads and targets, respectively
        agents_on_grid = state.grid[jax.vmap(tuple)(state.agents.position)]
        targets_on_grid = state.grid[jax.vmap(tuple)(state.agents.target)]
        assert (agents_on_grid == jnp.array([get_position(i) for i in range(3)])).all()
        assert (targets_on_grid == jnp.array([get_target(i) for i in range(3)])).all()

    def test_random_walk_generator__no_retrace(
        self,
        random_walk_generator: RandomWalkGenerator,
        key: chex.PRNGKey,
    ) -> None:
        """Checks that generator only traces the function once and works when jitted."""
        keys = jax.random.split(key, 2)
        jitted_generator = jax.jit(chex.assert_max_traces((random_walk_generator.__call__), n=1))

        for key in keys:
            jitted_generator(key)

    @staticmethod
    @pytest.mark.parametrize(
        ("function_input", "expected_value"),
        [
            (
                (
                    key,
                    (
                        valid_solved_grid_2,
                        generate_board_agents,
                        valid_training_grid,
                    ),
                )
            ),
        ],
    )
    def test_generate_board(
        random_walk_generator: RandomWalkGenerator,
        function_input: chex.PRNGKey,
        expected_value: Tuple[chex.Array, chex.Array, chex.Array],
    ) -> None:
        expected_solved_grid, expected_agents, expected_training_grid = expected_value
        solved_grid, agents, training_grid = random_walk_generator.generate_board(function_input)
        assert (training_grid == expected_training_grid).all()
        assert (solved_grid == expected_solved_grid).all()
        assert agents == expected_agents

    def test_generate_board_for_various_keys(
        self,
        random_walk_generator: RandomWalkGenerator,
    ) -> None:
        boards_generated = []
        number_of_keys_to_test = 10
        for i in range(number_of_keys_to_test):
            _, _, board = random_walk_generator.generate_board(jax.random.PRNGKey(i))
            boards_generated.append(board)

        for _i in range(number_of_keys_to_test):
            board = boards_generated.pop()
            for j in range(len(boards_generated)):
                assert not (board == boards_generated[j]).all()

    @staticmethod
    @pytest.mark.parametrize(
        ("function_input", "expected_value"),
        [
            (
                (key, valid_starting_grid, agents_starting),
                (
                    valid_starting_grid_after_1_step,
                    agents_starting_move_after_1_step,
                ),
            ),  # empty position
        ],
    )
    def test_step(
        random_walk_generator: RandomWalkGenerator,
        function_input: Tuple[chex.PRNGKey, chex.Array, Agent],
        expected_value: Tuple[chex.Array, Agent],
    ) -> None:
        agents_action_mask_after_1_step = get_action_masks(
            agents_starting_move_after_1_step, valid_starting_grid_after_1_step
        )
        expected_end_grid, expected_end_agents = expected_value
        expected_end_action_mask = get_action_masks(expected_end_agents, expected_end_grid)
        _, new_grid, new_agents, new_action_mask = random_walk_generator._step(
            (*function_input, agents_action_mask_after_1_step)
        )
        assert new_agents == expected_end_agents
        assert (new_grid == expected_end_grid).all()
        assert (new_action_mask == expected_end_action_mask).all()

    @staticmethod
    def test_initialize_agents(random_walk_generator: RandomWalkGenerator) -> None:
        grid, agents = random_walk_generator._initialize_agents(key, 5)
        assert agents == agents_starting_initialise_agents
        assert (grid == valid_starting_grid_initialize_agents).all()

    @staticmethod
    @pytest.mark.parametrize(
        ("function_input", "expected_value"),
        [
            (action_mask_all_valid, True),
            (action_mask_none_valid, False),
        ],
    )
    def test_continue_stepping(
        random_walk_generator: RandomWalkGenerator,
        function_input: chex.Array,
        expected_value: bool,
    ) -> None:
        continue_stepping = random_walk_generator._continue_stepping(
            (None, None, None, function_input)  # type: ignore
        )
        assert continue_stepping == expected_value

    @staticmethod
    def test_calculate_action_probs() -> None:
        action_mask = jnp.array([1, 1, 1, 1], dtype=jnp.bool_)
        actual = RandomWalkGenerator._calculate_action_probs(
            jnp.array([1, 1]), jnp.array([0, 0]), action_mask, 1.0
        )
        expected = jnp.array([0.06, 0.44, 0.44, 0.06])
        assert jnp.allclose(actual, expected, atol=1e-2)

    @staticmethod
    def test_calculate_action_probs_with_temperature() -> None:
        action_mask = jnp.array([1, 1, 1, 0], dtype=jnp.bool_)
        actual = RandomWalkGenerator._calculate_action_probs(
            jnp.array([1, 1]), jnp.array([0, 0]), action_mask, 2.0
        )
        expected = jnp.array([0.16, 0.42, 0.42, 0.0])
        assert jnp.allclose(actual, expected, atol=1e-2)
