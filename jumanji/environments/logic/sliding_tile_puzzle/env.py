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

from typing import Optional, Sequence, Tuple, cast

import chex
import jax
import jax.numpy as jnp
import matplotlib.animation as animation
from jax import lax
from numpy.typing import NDArray

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.logic.sliding_tile_puzzle.generator import (
    Generator,
    RandomGenerator,
)
from jumanji.environments.logic.sliding_tile_puzzle.reward import (
    DenseRewardFn,
    RewardFn,
)
from jumanji.environments.logic.sliding_tile_puzzle.types import Observation, State
from jumanji.environments.logic.sliding_tile_puzzle.viewer import (
    SlidingTilePuzzleViewer,
)
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class SlidingTilePuzzle(Environment[State]):
    """Environment for the Sliding Tile Puzzle problem.

    The problem is a combinatorial optimization task where the goal is
    to move the empty tile around in order to arrange all the tiles in order.
    See more info: https://en.wikipedia.org/wiki/Sliding_puzzle.

    - observation: `Observation`
        - puzzle: jax array (int32) of shape (N, N), representing the current state of the puzzle.
        - empty_tile_position: Tuple of int32, representing the position of the empty tile.
        - action_mask: jax array (bool) of shape (4,), indicating which actions are valid
            in the current state of the environment.

    - action: int32, representing the direction to move the empty tile
        (up, down, left, right)

    - reward: float, a dense reward is provided based on the arrangement of the tiles.
        It equals the negative sum of the boolean difference between
        the current state of the puzzle and the goal state (correctly arranged tiles).
        Each incorrectly placed tile contributes -1 to the reward.

    - episode termination: if the puzzle is solved.

    - state: `State`
        - puzzle: jax array (int32) of shape (N, N), representing the current state of the puzzle.
        - empty_tile_position: Tuple of int32, representing the position of the empty tile.
        - key: jax array (uint32) of shape (2,), random key used to generate random numbers
            at each step and for auto-reset.
    """

    def __init__(
        self,
        generator: Optional[Generator] = None,
        reward_fn: Optional[RewardFn] = None,
        viewer: Optional[Viewer[State]] = None,
    ) -> None:
        """Instantiate a `SlidingTilePuzzle` environment.

        Args:
            generator: callable to instantiate environment instances.
                Defaults to `RandomGenerator` which generates puzzles with
                a size of 5x5.
            reward_fn: RewardFn whose `__call__` method computes the reward of an environment
                transition. The function must compute the reward based on the current state,
                the chosen action and the next state.
                Implemented options are [`DenseRewardFn`, `SparseRewardFn`].
                Defaults to `DenseRewardFn`.
            viewer: environment viewer for rendering.
        """
        self.generator = generator or RandomGenerator(grid_size=5)
        self.reward_fn = reward_fn or DenseRewardFn()

        # Create viewer used for rendering
        self._env_viewer = viewer or SlidingTilePuzzleViewer(name="SlidingTilePuzzle")
        self.movements = jnp.array(
            [[-1, 0], [1, 0], [0, -1], [0, 1]]  # Up  # Down  # Left  # Right
        )
        self.solved_puzzle = jnp.arange(self.generator.grid_size**2).reshape(
            (self.generator.grid_size, self.generator.grid_size)
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment to an initial state."""
        key, subkey = jax.random.split(key)
        puzzle, empty_tile_position = self.generator(subkey)
        state = State(
            puzzle=puzzle,
            empty_tile_position=empty_tile_position,
            key=key,
        )
        action_mask = self._get_valid_actions(empty_tile_position)
        obs = Observation(
            puzzle=puzzle,
            empty_tile_position=empty_tile_position,
            action_mask=action_mask,
        )
        timestep = restart(observation=obs)
        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        """Updates the environment state after the agent takes an action."""
        (updated_puzzle, updated_empty_tile_position) = self._move_empty_tile(
            state.puzzle, state.empty_tile_position, action
        )
        # Check if the puzzle is solved
        done = jnp.array_equal(updated_puzzle, self.solved_puzzle)

        # Update the action mask
        action_mask = self._get_valid_actions(updated_empty_tile_position)

        next_state = State(
            puzzle=updated_puzzle,
            empty_tile_position=updated_empty_tile_position,
            key=state.key,
        )
        obs = Observation(
            puzzle=updated_puzzle,
            empty_tile_position=updated_empty_tile_position,
            action_mask=action_mask,
        )

        reward = self.reward_fn(state, action, next_state, self.solved_puzzle)

        timestep = lax.cond(done, termination, transition, reward, obs)

        return next_state, timestep

    def _move_empty_tile(
        self,
        puzzle: chex.Array,
        empty_tile_position: chex.Array,
        action: chex.Array,
    ) -> chex.Array:
        """Moves the empty tile in the given direction and returns the updated puzzle and reward."""

        # Compute the new position
        new_empty_tile_position = empty_tile_position + self.movements[action]

        # Predicate for the conditional
        is_valid_move = jnp.all(
            (new_empty_tile_position >= 0)
            & (new_empty_tile_position < self.generator.grid_size)
        )

        def valid_move(puzzle: chex.Array) -> Tuple[chex.Array, Tuple[int, int]]:
            # Swap the empty tile and the tile at the new position
            updated_puzzle = puzzle.at[tuple(empty_tile_position)].set(
                puzzle[tuple(new_empty_tile_position)]
            )
            updated_puzzle = updated_puzzle.at[tuple(new_empty_tile_position)].set(0)

            return updated_puzzle, new_empty_tile_position

        def invalid_move(puzzle: chex.Array) -> Tuple[chex.Array, Tuple[int, int]]:
            # If the move is not valid, return the original puzzle
            return puzzle, empty_tile_position

        return lax.cond(is_valid_move, valid_move, invalid_move, puzzle)

    def _get_valid_actions(self, empty_tile_position: chex.Array) -> chex.Array:
        # Compute the new positions if these movements are applied
        new_positions = empty_tile_position + self.movements

        # Check if the new positions are within the grid boundaries
        valid_moves_mask = jnp.all(
            (new_positions >= 0) & (new_positions < self.generator.grid_size), axis=-1
        )

        return valid_moves_mask

    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec."""
        n = self.generator.grid_size
        return specs.Spec(
            Observation,
            "ObservationSpec",
            puzzle=specs.BoundedArray(
                shape=(n, n),
                dtype=jnp.int32,
                minimum=-1,
                maximum=n * n - 1,
                name="puzzle",
            ),
            empty_tile_position=specs.BoundedArray(
                shape=(2,),
                dtype=jnp.int32,
                minimum=0,
                maximum=n - 1,
                name="empty_tile_position",
            ),
            action_mask=specs.BoundedArray(
                shape=(4,),
                dtype=bool,
                minimum=False,
                maximum=True,
                name="action_mask",
            ),
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec."""
        return specs.DiscreteArray(
            num_values=4, name="action", dtype=jnp.int32
        )  # Up, Down, Left, Right

    def render(self, state: State) -> Optional[NDArray]:
        """Renders the current state of the puzzle board.

        Args:
            state: is the current game state to be rendered.
        """
        return self._env_viewer.render(state=state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> animation.FuncAnimation:
        """Creates an animated gif of the puzzle board based on the sequence of game states.

        Args:
            states: is a list of `State` objects representing the sequence of game states.
            interval: the delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
            will not be stored.

        Returns:
            animation.FuncAnimation: the animation object that was created.
        """
        return self._env_viewer.animate(
            states=states, interval=interval, save_path=save_path
        )

    def close(self) -> None:
        """Perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        self._env_viewer.close()
