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

from typing import Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib
from numpy.typing import NDArray

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.packing.jigsaw.generator import (
    InstanceGenerator,
    RandomJigsawGenerator,
)
from jumanji.environments.packing.jigsaw.reward import DenseReward, RewardFn
from jumanji.environments.packing.jigsaw.types import Observation, State
from jumanji.environments.packing.jigsaw.utils import compute_grid_dim, rotate_piece
from jumanji.environments.packing.jigsaw.viewer import JigsawViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class Jigsaw(Environment[State]):

    """A Jigsaw solving environment with a configurable number of row and column pieces.
        Here the goal of an agent is to solve a jigsaw puzzle by placing pieces
        in their correct positions.

    - observation: Observation
        - current_board: jax array (float) of shape (num_rows, num_cols) with the
            current state of board.
        - pieces: jax array (float) of shape (num_pieces, 3, 3) with the pieces to
            be placed on the board. Here each piece is a 2D array with shape (3, 3).
        - action_mask: jax array (float) showing where which pieces can be placed on the board.
            this mask include all possible rotations and possible placement locations
            for each piece on the board.

    - action: jax array (int32) of shape ()
        multi discrete array containing the move to perform
        (piece to place, number of rotations, row coordinate, column coordinate).

    - reward: jax array (float) of shape (), could be either:
        - dense: the number of cells in the placed piece the overlaps with the correctly
            piece. this will be a value in the range [0, 9].
        - sparse: 1 if the board is solved, otherwise 0 at each timestep.

    - episode termination:
        - if all pieces have been placed on the board.
        - if the agent has taken `num_pieces` steps in the environment.

    - state: `State`
        - row_nibs_idxs: jax array (float) array containing row indices
            for selecting piece nibs during board generation.
        - col_nibs_idxs: jax array (float) array containing column indices
            for selecting piece nibs during board generation.
        - num_pieces: jax array (float) of shape () with the
            number of pieces in the jigsaw puzzle.
        - solved_board: jax array (float) of shape (num_rows, num_cols) with the
            solved board state.
        - pieces: jax array (float) of shape (num_pieces, 3, 3) with the pieces to
            be placed on the board.
        - action_mask: jax array (float) of shape (num_pieces, 4, num_rows, num_cols)
            showing where which pieces can be placed where on the board.
        - placed_pieces: jax array (bool) of shape (num_pieces,) showing which pieces
            have been placed on the board.
        - current_board: jax array (float) of shape (num_rows, num_cols) with the
            current state of board.
        - step_count: jax array (float) of shape () with the number of steps taken
            in the environment.
        - key: jax array (float) of shape (2,) with the random key used for board
            generation.

    ```python
    from jumanji.environments import Jigsaw
    env = Jigsaw()
    key = jax.random.key(0)
    state, timestep = jax.jit(env.reset)(key)
    env.render(state)
    action = env.action_spec().generate_value()
    state, timestep = jax.jit(env.step)(state, action)
    env.render(state)
    ```
    """

    def __init__(
        self,
        generator: Optional[InstanceGenerator] = None,
        reward_fn: Optional[RewardFn] = None,
        viewer: Optional[Viewer[State]] = None,
    ):
        """Initializes the environment.

        Args:
            generator: Instance generator for the environment.
        """

        default_generator = RandomJigsawGenerator(
            num_row_pieces=3,
            num_col_pieces=3,
        )

        self.generator = generator or default_generator
        self.num_row_pieces = self.generator.num_row_pieces
        self.num_col_pieces = self.generator.num_col_pieces
        self.num_pieces = self.num_row_pieces * self.num_col_pieces
        self.num_rows, self.num_cols = (
            compute_grid_dim(self.num_row_pieces),
            compute_grid_dim(self.num_col_pieces),
        )
        self.reward_fn = reward_fn or DenseReward()
        self.viewer = viewer or JigsawViewer(
            "Jigsaw", self.num_pieces, render_mode="human"
        )

    def __repr__(self) -> str:
        return (
            f"Jigsaw environment with a puzzle size of ({self.num_rows}x{self.num_cols}) "
            f"with {self.num_row_pieces} row pieces, {self.num_col_pieces} column "
            f"pieces. Each piece has dimension (3x3)."
        )

    def reset(
        self,
        key: chex.PRNGKey,
    ) -> Tuple[State, TimeStep[Observation]]:

        """Resets the environment.

        Args:
            key: PRNG key for generating a new instance.

        Returns:
            a tuple of the initial state and a time step.
        """

        board_state = self.generator(key)

        obs = self._observation_from_state(board_state)
        timestep = restart(observation=obs)

        return board_state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        """Steps the environment.

        Args:
            state: current state of the environment.
            action: action to take.

        Returns:
            a tuple of the next state and a time step.
        """
        # Unpack and use actions
        piece_idx, rotation, row_idx, col_idx = action

        chosen_piece = state.pieces[piece_idx]

        # Rotate chosen piece
        chosen_piece = rotate_piece(chosen_piece, rotation)

        grid_piece = self._expand_piece_to_board(chosen_piece, row_idx, col_idx)
        grid_mask_piece = self._get_ones_like_expanded_piece(grid_piece=grid_piece)

        action_is_legal = self._check_action_is_legal(
            action, state.current_board, state.placed_pieces, grid_mask_piece
        )

        # If the action is legal
        new_board = jax.lax.cond(
            action_is_legal,
            lambda: state.current_board + grid_piece,
            lambda: state.current_board,
        )
        placed_pieces = jax.lax.cond(
            action_is_legal,
            lambda: state.placed_pieces.at[piece_idx].set(True),
            lambda: state.placed_pieces,
        )

        new_action_mask = self._make_full_action_mask(
            new_board, state.pieces, placed_pieces
        )

        next_state = State(
            col_nibs_idxs=state.col_nibs_idxs,
            row_nibs_idxs=state.row_nibs_idxs,
            solved_board=state.solved_board,
            current_board=new_board,
            pieces=state.pieces,
            action_mask=new_action_mask,
            num_pieces=state.num_pieces,
            key=state.key,
            step_count=state.step_count + 1,
            placed_pieces=placed_pieces,
        )

        done = self._check_done(next_state)

        next_obs = self._observation_from_state(next_state)

        reward = self.reward_fn(state, grid_piece, next_state, action_is_legal, done)

        timestep = jax.lax.cond(
            done,
            termination,
            transition,
            reward,
            next_obs,
        )

        return next_state, timestep

    def render(self, state: State) -> Optional[NDArray]:
        """Render a given state of the environment.

        Args:
            state: `State` object containing the current environment state.
        """
        return self.viewer.render(state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of states.

        Args:
            states: sequence of `State` corresponding to subsequent timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            animation that can export to gif, mp4, or render with HTML.
        """
        return self.viewer.animate(states, interval, save_path)

    def close(self) -> None:
        """Perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        self.viewer.close()

    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec of the environment.

        Returns:
            Spec for each filed in the observation:
            - current_board: BoundedArray (int) of shape (num_rows, num_cols).
            - pieces: BoundedArray (int) of shape (num_pieces, 3, 3).
            - action_mask: BoundedArray (bool) of shape (num_pieces,).
        """

        current_board = specs.BoundedArray(
            shape=(self.num_rows, self.num_cols),
            minimum=0,
            maximum=self.num_pieces,
            dtype=jnp.float32,
            name="current_board",
        )

        pieces = specs.BoundedArray(
            shape=(self.num_pieces, 3, 3),
            minimum=0,
            maximum=self.num_pieces,
            dtype=jnp.float32,
            name="pieces",
        )

        action_mask = specs.BoundedArray(
            shape=(
                self.num_pieces,
                4,
                self.num_rows - 2,
                self.num_cols - 2,
            ),
            minimum=False,
            maximum=True,
            dtype=bool,
            name="action_mask",
        )

        return specs.Spec(
            Observation,
            "ObservationSpec",
            current_board=current_board,
            pieces=pieces,
            action_mask=action_mask,
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Specifications of the action expected by the `JigSaw` environment.

        Returns:
            MultiDiscreteArray (int32) of shape (num_pieces, num_rotations,
            max_row_position, max_col_position).
            - num_pieces: int between 0 and num_pieces - 1 (included).
            - num_rotations: int between 0 and 3 (included).
            - max_row_position: int between 0 and max_row_position - 1 (included).
            - max_col_position: int between 0 and max_col_position - 1 (included).
        """

        max_row_position = self.num_rows - 2
        max_col_position = self.num_cols - 2

        return specs.MultiDiscreteArray(
            num_values=jnp.array(
                [self.num_pieces, 4, max_row_position, max_col_position]
            ),
            name="action",
        )

    def _check_done(self, state: State) -> bool:
        """Checks if the environment is done by checking whether the number of
            steps is equal to the number of pieces in the puzzle.

        Args:
            state: current state of the environment.

        Returns:
            True if the environment is done, False otherwise.
        """

        done: bool = state.step_count >= state.num_pieces

        return done

    def _check_action_is_legal(
        self,
        action: chex.Numeric,
        current_board: chex.Array,
        placed_pieces: chex.Array,
        grid_mask_piece: chex.Array,
    ) -> bool:
        """Checks if the action is legal by considering the action mask and the
            board mask. An action is legal if the action mask is True for that action
            and the board mask indicates that there is no overlap with pieces
            already placed.

        Args:
            action: action taken.
            state: current state of the environment.
            grid_mask_piece: grid with ones where the piece is placed.

        Returns:
            True if the action is legal, False otherwise.
        """

        piece_idx, _, _, _ = action

        placed_mask = (current_board > 0.0) + grid_mask_piece

        legal: bool = (~placed_pieces[piece_idx]) & (jnp.max(placed_mask) <= 1)

        return legal

    def _get_ones_like_expanded_piece(self, grid_piece: chex.Array) -> chex.Array:
        """Makes a grid of zeroes with ones where the piece is placed.

        Args:
            grid_piece: piece placed on a grid of zeroes.
        """

        grid_with_ones = jnp.where(grid_piece != 0, 1, 0)

        return grid_with_ones

    def _expand_piece_to_board(
        self,
        piece: chex.Array,
        row_coord: chex.Numeric,
        col_coord: chex.Numeric,
    ) -> chex.Array:
        """Takes a piece and places it on a grid of zeroes with the same size as the board.

        Args:
            state: current state of the environment.
            piece: piece to place on the board.
            row_coord: row coordinate on the board where the top left corner
                of the piece will be placed.
            col_coord: column coordinate on the board where the top left corner
                of the piece will be placed.

        Returns:
            Grid of zeroes with values where the piece is placed.
        """

        grid_with_piece = jnp.zeros((self.num_rows, self.num_cols), dtype=jnp.float32)
        place_location = (row_coord, col_coord)

        grid_with_piece = jax.lax.dynamic_update_slice(
            grid_with_piece, piece, place_location
        )

        return grid_with_piece

    def _observation_from_state(self, state: State) -> Observation:
        """Creates an observation from a state.

        Args:
            state: State to create an observation from.

        Returns:
            An observation.
        """

        return Observation(
            current_board=state.current_board,
            action_mask=state.action_mask,
            pieces=state.pieces,
        )

    def _expand_all_pieces_to_boards(
        self,
        pieces: chex.Array,
        piece_idxs: chex.Array,
        rotations: chex.Array,
        rows: chex.Array,
        cols: chex.Array,
    ) -> chex.Array:
        """Takes multiple pieces and their corresponding rotations and positions,
            and generates a grid for each piece.

        Args:
            pieces: array of possible pieces.
            piece_idxs: array of indices of the pieces to place.
            rotations: array of all possible rotations for each piece.
            rows: array of row coordinates.
            cols: array of column coordinates.
        """

        batch_expand_piece_to_board = jax.vmap(
            self._expand_piece_to_board, in_axes=(0, 0, 0)
        )

        all_possible_pieces = pieces[piece_idxs]
        rotated_pieces = jax.vmap(rotate_piece, in_axes=(0, 0))(
            all_possible_pieces, rotations
        )
        grids = batch_expand_piece_to_board(rotated_pieces, rows, cols)

        batch_get_ones_like_expanded_piece = jax.vmap(
            self._get_ones_like_expanded_piece, in_axes=(0)
        )
        grids = batch_get_ones_like_expanded_piece(grids)
        return grids

    def _make_full_action_mask(
        self, current_board: chex.Array, pieces: chex.Array, placed_pieces: chex.Array
    ) -> chex.Array:
        """Create a mask of possible actions based on the current state of the board.

        Args:
            current_board: current state of the board.
            pieces: array of all pieces.
            placed_pieces: array of pieces that have already been placed.
        """
        num_pieces, num_rotations, num_rows, num_cols = (
            self.num_pieces,
            4,
            self.num_rows - 2,
            self.num_cols - 2,
        )

        pieces_grid, rotations_grid, rows_grid, cols_grid = jnp.meshgrid(
            jnp.arange(num_pieces),
            jnp.arange(num_rotations),
            jnp.arange(num_rows),
            jnp.arange(num_cols),
        )

        grid_mask_pieces = self._expand_all_pieces_to_boards(
            pieces,
            pieces_grid.flatten(),
            rotations_grid.flatten(),
            rows_grid.flatten(),
            cols_grid.flatten(),
        )

        batch_check_action_is_legal = jax.vmap(
            self._check_action_is_legal, in_axes=(0, None, None, 0)
        )
        legal_actions = batch_check_action_is_legal(
            jnp.stack(
                (pieces_grid, rotations_grid, rows_grid, cols_grid), axis=-1
            ).reshape(-1, 4),
            current_board,
            placed_pieces,
            grid_mask_pieces,
        )

        legal_actions = legal_actions.reshape(
            num_pieces, num_rotations, num_rows, num_cols
        )

        # Now set all current placed pieces to false in the mask.
        placed_pieces_array = placed_pieces.reshape((self.num_pieces, 1, 1, 1))
        placed_pieces_mask = jnp.tile(
            placed_pieces_array, (1, num_rotations, num_rows, num_cols)
        )
        legal_actions = jnp.where(placed_pieces_mask, False, legal_actions)

        return legal_actions
