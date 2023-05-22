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

    """A Jigsaw solving environment."""

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
            num_row_pieces=5,
            num_col_pieces=5,
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
        self, key: chex.PRNGKey, generate_new_board: bool = False
    ) -> Tuple[State, TimeStep[Observation]]:

        """Resets the environment.

        Args:
            key: PRNG key for generating a new instance.
            generate_new_board: whether to generate a new board
                or reset the current one.

        Returns:
            a tuple of the initial state and a time step.
        """

        board_state = self.generator(key)

        board_state.action_mask = jnp.ones(self.num_pieces, dtype=bool)
        board_state.current_board = jnp.zeros_like(board_state.solved_board)

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

        grid_piece = self._expand_piece_to_board(state, chosen_piece, row_idx, col_idx)

        grid_mask_piece = self._get_ones_like_expanded_piece(grid_piece=grid_piece)

        action_is_legal = self._check_action_is_legal(piece_idx, state, grid_mask_piece)

        next_state_legal = State(
            col_nibs_idxs=state.col_nibs_idxs,
            row_nibs_idxs=state.row_nibs_idxs,
            solved_board=state.solved_board,
            current_board=state.current_board + grid_piece,
            pieces=state.pieces,
            action_mask=state.action_mask.at[piece_idx].set(False),
            num_pieces=state.num_pieces,
            key=state.key,
            step_count=state.step_count + 1,
        )

        next_state_illegal = State(
            col_nibs_idxs=state.col_nibs_idxs,
            row_nibs_idxs=state.row_nibs_idxs,
            solved_board=state.solved_board,
            current_board=state.current_board,
            pieces=state.pieces,
            action_mask=state.action_mask,
            num_pieces=state.num_pieces,
            key=state.key,
            step_count=state.step_count + 1,
        )

        # Transition board to new state if the action is legal
        # otherwise, stay in the same state.
        next_state = jax.lax.cond(
            action_is_legal,
            lambda: next_state_legal,
            lambda: next_state_illegal,
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
            - current_board: BoundedArray (int) of shape (board_dim[0], board_dim[1]).
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
            shape=(self.num_pieces,),
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

        max_row_position = self.num_rows - 3
        max_col_position = self.num_cols - 3

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
        self, action: chex.Numeric, state: State, grid_mask_piece: chex.Array
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

        placed_mask = (state.current_board > 0.0) + grid_mask_piece

        legal: bool = state.action_mask[action] & (jnp.max(placed_mask) <= 1)

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
        state: State,
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

        grid_with_piece = jnp.zeros_like(state.solved_board)

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
