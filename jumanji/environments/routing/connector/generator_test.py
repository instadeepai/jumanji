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
from jumanji.environments.routing.connector.generator_test_expected_outputs import (
    agents_finished,
    agents_reshaped_for_generator,
    agents_starting,
    agents_starting_move_1_step_up,
    empty_grid,
    grid_to_test_available_cells,
    grids_after_1_agent_step,
    key,
    key_2,
    valid_end_grid,
    valid_end_grid2,
    valid_starting_grid,
    valid_starting_grid_after_1_step,
)
from jumanji.environments.routing.connector.types import Agent
from jumanji.environments.routing.connector.utils import get_position, get_target


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
    jitted_generator = jax.jit(
        chex.assert_max_traces((uniform_random_generator.__call__), n=1)
    )

    for key in keys:
        jitted_generator(key)


class TesRandomWalkGenerator:
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
        jitted_generator = jax.jit(
            chex.assert_max_traces((random_walk_generator.__call__), n=1)
        )

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
                        agents_reshaped_for_generator.start,
                        agents_reshaped_for_generator.position,
                        valid_end_grid2,
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
        expected_heads, expected_targets, expected_grid = expected_value
        heads, targets, grid = random_walk_generator.generate_board(key)
        assert (grid == expected_grid).all()
        assert (heads == expected_heads).all()
        assert (targets == expected_targets).all()

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
                    key_2,
                    valid_starting_grid_after_1_step,
                    agents_starting_move_1_step_up,
                ),
            ),  # empty position
        ],
    )
    def test_step(
        random_walk_generator: RandomWalkGenerator,
        function_input: Tuple[chex.PRNGKey, chex.Array, Agent],
        expected_value: Tuple[chex.PRNGKey, chex.Array, Agent],
    ) -> None:
        end_key, end_grid, end_agents = expected_value

        new_key, new_grid, new_agents = random_walk_generator._step(function_input)
        assert new_agents == end_agents
        assert (new_grid == end_grid).all()
        assert (new_key == end_key).all()

    def test_initialise_agents(
        self, random_walk_generator: RandomWalkGenerator
    ) -> None:
        grid, agents = random_walk_generator._initialise_agents(key, empty_grid)
        assert agents == agents_starting
        assert (grid == valid_starting_grid).all()

    def test_place_agent_heads_on_grid(
        self,
        random_walk_generator: RandomWalkGenerator,
    ) -> None:
        expected_output = jnp.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 5.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 8.0, 0.0, 0.0],
                ],
            ],
        )

        grid_per_agent = jax.vmap(
            random_walk_generator._place_agent_heads_on_grid, in_axes=(None, 0)
        )(empty_grid, agents_starting)
        assert (grid_per_agent == expected_output).all()

    @staticmethod
    @pytest.mark.parametrize(
        ("function_input", "expected_value"),
        [
            ((key, valid_end_grid, agents_finished), False),
            ((key, valid_starting_grid, agents_starting), True),
        ],
    )
    def test_continue_stepping(
        random_walk_generator: RandomWalkGenerator,
        function_input: Tuple[chex.PRNGKey, chex.Array, Agent],
        expected_value: bool,
    ) -> None:
        continue_stepping = random_walk_generator._continue_stepping(function_input)
        assert continue_stepping == expected_value

    @staticmethod
    @pytest.mark.parametrize(
        ("function_input", "expected_value"),
        [
            ((valid_end_grid, agents_finished), jnp.array([True, True, True])),
            ((valid_starting_grid, agents_starting), jnp.array([False, False, False])),
        ],
    )
    def test_no_available_cells(
        random_walk_generator: RandomWalkGenerator,
        function_input: Tuple[chex.Array, Agent],
        expected_value: chex.Array,
    ) -> None:
        grid, agents = function_input
        dones = jax.vmap(random_walk_generator._no_available_cells, in_axes=(None, 0))(
            grid, agents
        )
        assert (dones == expected_value).all()

    @staticmethod
    @pytest.mark.parametrize(
        ("function_input", "expected_value"),
        [
            (jnp.array(7), jnp.array([1, 2])),
            (jnp.array(24), jnp.array([4, 4])),  # corner
            (jnp.array(1), jnp.array([0, 1])),  # edge
        ],
    )
    def test_convert_flat_position_to_tuple(
        random_walk_generator: RandomWalkGenerator,
        function_input: chex.Array,
        expected_value: chex.Array,
    ) -> None:
        position_tuple = random_walk_generator._convert_flat_position_to_tuple(
            function_input
        )
        assert (position_tuple == expected_value).all()

    @staticmethod
    @pytest.mark.parametrize(
        ("function_input", "expected_value"),
        [
            (jnp.array([1, 2]), jnp.array(7)),
            (jnp.array([4, 4]), jnp.array(24)),  # corner
            (jnp.array([0, 1]), jnp.array(1)),  # edge
        ],
    )
    def test_convert_tuple_to_flat_position(
        random_walk_generator: RandomWalkGenerator,
        function_input: chex.Array,
        expected_value: chex.Array,
    ) -> None:
        position_tuple = random_walk_generator._convert_tuple_to_flat_position(
            function_input
        )
        assert (position_tuple == expected_value).all()

    @staticmethod
    @pytest.mark.parametrize(
        ("function_input", "expected_value"),
        [
            ((0, 1), 2),  # move from 0 to 1 (right)
            ((1, 0), 4),  # move from 1 to 0 (left)
            ((0, 5), 3),  # move from 0 to 5 (down)
            ((5, 0), 1),  # move from 5 to 0 (up)
        ],
    )
    def test_action_from_position(
        random_walk_generator: RandomWalkGenerator,
        function_input: Tuple[chex.Array, chex.Array],
        expected_value: chex.Array,
    ) -> None:
        position_1, position_2 = function_input
        action = random_walk_generator._action_from_positions(position_1, position_2)
        assert action == expected_value

    @staticmethod
    @pytest.mark.parametrize(
        ("function_input", "expected_value"),
        [
            (jnp.array([1, 0]), 3),  # down
            (jnp.array([-1, 0]), 1),  # up
            (jnp.array([0, -1]), 4),  # left
            (jnp.array([0, 1]), 2),  # right
            (jnp.array([0, 0]), 0),  # none
        ],
    )
    def test_action_from_tuple(
        random_walk_generator: RandomWalkGenerator,
        function_input: chex.Array,
        expected_value: chex.Array,
    ) -> None:
        action = random_walk_generator._action_from_tuple(function_input)
        assert action == expected_value

    @staticmethod
    @pytest.mark.parametrize(
        ("function_input", "expected_value"),
        [
            ((0), jnp.array([-1, 5, -1, 1])),
            (
                (6),
                jnp.array([1, 11, 5, 7]),
            ),  # adjacent cells in order up, down, left, right
        ],
    )
    def test_adjacent_cells(
        random_walk_generator: RandomWalkGenerator,
        function_input: int,
        expected_value: chex.Array,
    ) -> None:
        adjacent_cells = random_walk_generator._adjacent_cells(function_input)
        assert (adjacent_cells == expected_value).all()

    @staticmethod
    @pytest.mark.parametrize(
        ("function_input", "expected_value"),
        [
            ((valid_end_grid, 8), jnp.array([-1, -1, -1, -1])),
            ((grid_to_test_available_cells, 8), jnp.array([-1, 13, -1, -1])),
            # ((6), jnp.array([1, 11, 5, 7])), #adjacent cells in order up, down, left, right
        ],
    )
    def test_available_cells(
        random_walk_generator: RandomWalkGenerator,
        function_input: Tuple[chex.Array, chex.Array],
        expected_value: chex.Array,
    ) -> None:
        grid_1, cell = function_input
        available_cells = random_walk_generator._available_cells(grid_1, cell)
        assert (available_cells == expected_value).all()

    @staticmethod
    @pytest.mark.parametrize(
        ("function_input", "expected_value"),
        [
            ((valid_starting_grid, 21), True),  # empty position
            ((valid_starting_grid, 22), False),  # taken position
            ((valid_starting_grid, 32), False),  # test position that is not in the grid
        ],
    )
    def test_is_cell_free(
        random_walk_generator: RandomWalkGenerator,
        function_input: Tuple[chex.Array, chex.Array],
        expected_value: bool,
    ) -> None:
        grid_1, cell = function_input
        is_cell_free = random_walk_generator._is_cell_free(grid_1, cell)
        assert is_cell_free == expected_value

    @staticmethod
    @pytest.mark.parametrize(
        ("function_input", "expected_value"),
        [
            (
                (agents_starting, valid_starting_grid, jnp.array([1, 1, 1])),
                (grids_after_1_agent_step, agents_starting_move_1_step_up),
            ),  # empty position
        ],
    )
    def test_step_agent(
        random_walk_generator: RandomWalkGenerator,
        function_input: Tuple[Agent, chex.Array, int],
        expected_value: Tuple[chex.Array, Agent],
    ) -> None:
        agent, grid, action = function_input
        expected_grids, expected_agents = expected_value
        new_agents, new_grids = jax.vmap(
            random_walk_generator._step_agent, in_axes=(0, None, 0)
        )(agent, grid, action)
        assert new_agents == expected_agents
        assert (new_grids == expected_grids).all()

    @staticmethod
    @pytest.mark.parametrize(
        ("function_input", "expected_value"),
        [
            (
                (valid_starting_grid, agents_starting, jnp.array([1, 2])),
                True,
            ),  # empty position
            ((valid_starting_grid, agents_starting, jnp.array([1, 1])), True),
            ((valid_starting_grid, agents_starting, jnp.array([-1, 1])), False),
        ],
    )
    def test_is_valid_position_rw(
        random_walk_generator: RandomWalkGenerator,
        function_input: Tuple[chex.Array, Agent, chex.Array],
        expected_value: chex.Array,
    ) -> None:
        grid, agent, new_position = function_input
        valid_position = random_walk_generator._is_valid_position(
            grid, agent, new_position
        )
        assert (valid_position == expected_value).all()
