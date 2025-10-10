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

import abc
from typing import Any, Tuple

import chex
import jax
import jax.nn
from jax import numpy as jnp

from jumanji.environments.routing.connector.constants import (
    PATH,
    POSITION,
)
from jumanji.environments.routing.connector.types import Agent, State
from jumanji.environments.routing.connector.utils import (
    get_action_masks,
    get_adjacency_mask,
    get_path,
    get_position,
    get_surrounded_mask,
    get_target,
    is_repeated_later,
    move_position,
)


class Generator(abc.ABC):
    """Base class for generators for the connector environment."""

    def __init__(self, grid_size: int, num_agents: int) -> None:
        """Initialises a connector generator, used to generate grids for the Connector environment.

        Args:
            grid_size: size of the grid to generate.
            num_agents: number of agents on the grid.
        """
        self._grid_size = grid_size
        self._num_agents = num_agents

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `Connector` state that contains the grid and the agents' layout.

        Returns:
            A `Connector` state.
        """


class UniformRandomGenerator(Generator):
    """Randomly generates `Connector` grids that may or may not be solvable. This generator places
    start and target positions uniformly at random on the grid.
    """

    def __init__(self, grid_size: int, num_agents: int) -> None:
        """Instantiates a `UniformRandomGenerator`.

        Args:
            grid_size: size of the square grid to generate.
            num_agents: number of agents/paths on the grid.
        """
        super().__init__(grid_size, num_agents)

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `Connector` state that contains the grid and the agents' layout.

        Returns:
            A `Connector` state.
        """
        key, pos_key = jax.random.split(key)
        starts_flat, targets_flat = jax.random.choice(
            key=pos_key,
            a=jnp.arange(self.grid_size**2),
            shape=(2, self.num_agents),  # Start and target positions for all agents
            replace=False,  # Start and target positions cannot overlap
        )

        # Create 2D points from the flat arrays.
        starts = jnp.divmod(starts_flat, self.grid_size)
        targets = jnp.divmod(targets_flat, self.grid_size)

        # Get the agent values for starts and positions.
        agent_position_values = jax.vmap(get_position)(jnp.arange(self.num_agents))
        agent_target_values = jax.vmap(get_target)(jnp.arange(self.num_agents))

        # Create empty grid.
        grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)

        # Place the agent values at starts and targets.
        grid = grid.at[starts].set(agent_position_values)
        grid = grid.at[targets].set(agent_target_values)

        # Create the agent pytree that corresponds to the grid.
        agents = jax.vmap(Agent)(
            id=jnp.arange(self.num_agents),
            start=jnp.stack(starts, axis=1),
            target=jnp.stack(targets, axis=1),
            position=jnp.stack(starts, axis=1),
        )

        step_count = jnp.array(0, jnp.int32)
        action_mask = get_action_masks(agents, grid)

        return State(
            key=key, grid=grid, step_count=step_count, agents=agents, action_mask=action_mask
        )


class RandomWalkGenerator(Generator):
    """Randomly generates `Connector` grids that are guaranteed be solvable.

    This generator places start positions randomly on the grid and performs a random walk from each.
    Targets are placed at their terminuses.
    """

    def __init__(
        self, grid_size: int, num_agents: int, temperature: float = 1.0, no_u_turn: bool = False
    ) -> None:
        """Instantiates a `RandomWalkGenerator.

        Args:
            grid_size: size of the square grid to generate.
            num_agents: number of agents/paths on the grid.
        """
        super().__init__(grid_size, num_agents)
        self.temperature = temperature
        self.no_u_turn = no_u_turn

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `Connector` state that contains the grid and the agents' layout.

        Args:
            key: used to randomly generate the connector grid.

        Returns:
            A `Connector` state.
        """
        key, board_key = jax.random.split(key)
        _, agents, grid = self.generate_board(board_key)
        step_count = jnp.array(0, jnp.int32)
        action_mask = get_action_masks(agents, grid)
        return State(
            key=key, grid=grid, step_count=step_count, agents=agents, action_mask=action_mask
        )

    def generate_board(self, key: chex.PRNGKey) -> Tuple[chex.Array, Agent, chex.Array]:
        """Generates solvable board using random walk.

        Args:
            key: random key.

        Returns:
            Tuple containing solved board, the agents and an empty training board.
        """
        key, step_key = jax.random.split(key)
        grid, agents, action_mask, last_two_actions = self._initialize(key, self.grid_size)

        stepping_tuple = (step_key, grid, agents, action_mask, last_two_actions)

        _, grid, agents, _, _ = jax.lax.while_loop(
            self._continue_stepping, self._step, stepping_tuple
        )

        # Convert heads and targets to format accepted by generator
        heads = agents.start.T
        targets = agents.position.T

        solved_grid = self.update_solved_board_with_head_target_encodings(
            grid, tuple(heads), tuple(targets)
        )
        # Update agent information to include targets and positions after first step
        agents.target = agents.position
        agents.position = agents.start

        agent_position_values = get_position(jnp.arange(self.num_agents))
        agent_target_values = get_target(jnp.arange(self.num_agents))
        # Populate an empty grid with heads and targets
        grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
        grid = grid.at[tuple(agents.start.T)].set(agent_position_values)
        grid = grid.at[tuple(agents.target.T)].set(agent_target_values)
        return solved_grid, agents, grid

    @staticmethod
    def _get_action_mask_no_u_turn(
        agents: Agent, grid: chex.Array, last_two_actions: chex.Array
    ) -> chex.Array:
        """Gets the action mask for all agents without allowing U-turns."""
        action_mask = get_action_masks(agents, grid)

        illegal_actions = jnp.array([0, 3, 4, 1, 2])
        illegal_actions_idx = illegal_actions[last_two_actions[:, 0]]
        illegal_actions = (
            jnp.zeros_like(action_mask)
            .at[jnp.arange(action_mask.shape[0]), illegal_actions_idx]
            .set(True)
            .at[:, 0]
            .set(False)
        )
        action_mask = action_mask & ~illegal_actions
        # jax.debug.breakpoint()
        return action_mask

    def _step(
        self, stepping_tuple: Tuple[chex.PRNGKey, chex.Array, Agent, chex.Array, chex.Array]
    ) -> Tuple[chex.PRNGKey, chex.Array, Agent, chex.Array, chex.Array]:
        """Steps all agents at the same time correcting for possible collisions.

        If a collision occurs we place the agent with the lower `agent_id` in its previous position.
        This method is equivalent in function to _step_agents from 'Connector' environment.

        Returns:
            Tuple of agents and grid after having applied each agents' action
        """
        key, grid, agents, action_mask, last_two_actions = stepping_tuple
        key, next_key = jax.random.split(key)

        agent_ids = jnp.arange(self.num_agents)
        keys = jax.random.split(key, num=self.num_agents)

        # Randomly select action for each agent
        actions = jax.vmap(self._select_action)(keys, action_mask, agents.start, agents.position)

        is_movement = actions != 0  # 0 is NOOP
        # Shift previous action to column 0, but only if current action is a movement
        new_col_0 = jnp.where(is_movement, last_two_actions[:, 1], last_two_actions[:, 0])
        # Set column 1 to current action only if it's a movement, otherwise keep old value
        new_col_1 = jnp.where(is_movement, actions, last_two_actions[:, 1])
        last_two_actions = jnp.stack([new_col_0, new_col_1], axis=1)

        new_positions = jax.vmap(move_position)(agents.position, actions)
        collided = is_repeated_later(new_positions)
        new_positions = jnp.where(collided[:, jnp.newaxis], agents.position, new_positions)

        noop = jnp.all(new_positions == agents.position, axis=-1)

        # Change old position from a POSITION to a PATH if not a NOOP
        old_position_values = (PATH - POSITION) * ~noop
        # Add 0 (no change) if doing a noop
        new_position_values = jax.vmap(get_position)(agent_ids) * ~noop

        grid = (
            # Set new values at previous position - likely change it to a PATH
            grid.at[tuple(agents.position.T)]
            .add(old_position_values, unique_indices=True)
            # Set new values at new position - likely change it to a POSITION
            .at[tuple(new_positions.T)]
            .add(new_position_values)  # not necessarily unique inds (if collided)
        )

        new_agents = agents.replace(position=new_positions)  # type: ignore
        if self.no_u_turn:
            new_action_mask = self._get_action_mask_no_u_turn(new_agents, grid, last_two_actions)
        else:
            new_action_mask = get_action_masks(new_agents, grid)

        return next_key, grid, new_agents, new_action_mask, last_two_actions

    def _initialize(
        self, key: chex.PRNGKey, grid_size: int
    ) -> Tuple[chex.Array, Agent, chex.Array, chex.Array]:
        """Initializes agents using random starting point and places heads on the grid.

        Args:
            key: random key.
            grid: empty grid.

        Returns:
            Tuple of grid with populated starting points and agents initialized with
            the same starting points.
        """
        grid = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
        # Generate locations of heads and an adjacent first move for each agent.
        # Return a grid with these positions populated.
        carry, heads_and_positions = jax.lax.scan(
            self._initialize_starts_and_first_move,
            (key, grid),
            jnp.arange(self.num_agents),
        )
        starts, first_move = heads_and_positions
        _, grid = carry

        # Fill target with default value as targets will be assigned after random walk
        targets = jnp.full((2, self.num_agents), -1)

        # # Initialize agents
        agents = jax.vmap(Agent)(
            id=jnp.arange(self.num_agents),
            start=jnp.stack(starts, axis=1),
            target=jnp.stack(targets, axis=1),
            position=jnp.stack(first_move, axis=1),
        )
        action_mask = get_action_masks(agents, grid)
        last_two_actions = jnp.zeros((self.num_agents, 2), dtype=jnp.int32)
        return grid, agents, action_mask, last_two_actions

    def _initialize_starts_and_first_move(
        self,
        carry: Tuple[chex.PRNGKey, chex.Array],
        agent_id: int,
    ) -> Tuple[Tuple[chex.PRNGKey, chex.Array], Tuple[chex.Array, chex.Array]]:
        """Simplified version that initializes starting positions and first move of each agent."""
        key, grid = carry
        not_occupied_mask = grid == 0
        not_surrounded_mask = ~get_surrounded_mask(grid)

        def get_random_valid_coordinate(
            key: chex.PRNGKey, mask: chex.Array
        ) -> Tuple[chex.Array, chex.Array]:
            """
            Randomly selects a single valid (True) coordinate from a boolean mask.
            """
            flat_mask = mask.flatten()

            # Randomly choose one of those flat indices
            random_choice_flat = jax.random.choice(key, len(flat_mask), p=flat_mask)

            # Convert the chosen flat index back to 2D grid coordinates
            random_coordinate = jnp.unravel_index(random_choice_flat, mask.shape)

            return random_coordinate

        start_key, first_move_key, next_key = jax.random.split(key, 3)
        start_coordinate = get_random_valid_coordinate(
            start_key, not_surrounded_mask & not_occupied_mask
        )
        grid = grid.at[start_coordinate].set(get_path(agent_id))

        adjacency_mask = get_adjacency_mask(grid.shape, start_coordinate)
        not_occupied_mask = grid == 0
        first_move_coordinate = get_random_valid_coordinate(
            first_move_key, adjacency_mask & not_occupied_mask
        )
        grid = grid.at[first_move_coordinate].set(get_position(agent_id))

        return (next_key, grid), (start_coordinate, first_move_coordinate)

    def _continue_stepping(
        self, stepping_tuple: Tuple[chex.PRNGKey, chex.Array, Agent, chex.Array, chex.Array]
    ) -> chex.Array:
        """Determines if agents can continue taking steps."""
        _, _, _, action_mask, _ = stepping_tuple

        return action_mask[:, 1:].any()

    def _select_action(
        self,
        key: chex.PRNGKey,
        action_mask: chex.Array,
        start_position: chex.Array,
        current_position: chex.Array,
    ) -> chex.Array:
        """Selects action for agent to take given its current position.

        Args:
            key: random key.
            action_mask: action mask for the agent.

        Returns:
            Integer corresponding to the action the agent will take in its next step.
            Action indices match those in connector.constants.
        """
        action_probs = self._calculate_action_probs(
            current_position, start_position, action_mask[1:], self.temperature
        )

        action = jax.random.choice(key=key, a=jnp.arange(1, 5), p=action_probs)
        can_move = action_mask[1:].any()
        action = action * can_move

        return action

    @staticmethod
    def _calculate_action_probs(
        current_position: chex.Array,
        start_position: chex.Array,
        action_mask: chex.Array,
        temperature: float,
    ) -> chex.Array:
        displacement = current_position - start_position
        action_dot_products = jnp.array(
            [-displacement[0], displacement[1], displacement[0], -displacement[1]]
        )
        action_probs = jax.nn.softmax(action_dot_products / temperature) * action_mask
        action_probs = action_probs / action_probs.sum()
        return action_probs

    def update_solved_board_with_head_target_encodings(
        self,
        solved_grid: chex.Array,
        heads: Tuple[Any, ...],
        targets: Tuple[Any, ...],
    ) -> chex.Array:
        """Updates grid array with all agent encodings."""
        agent_position_values = get_position(jnp.arange(self.num_agents))
        agent_target_values = get_target(jnp.arange(self.num_agents))
        # Transpose the agent_position_values to match the shape of the grid.
        # Place the agent values at starts and targets.
        solved_grid = solved_grid.at[heads].set(agent_position_values)
        solved_grid = solved_grid.at[targets].set(agent_target_values)
        return solved_grid
