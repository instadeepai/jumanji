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
from jumanji.environments.packing.flat_pack.generator import (
    InstanceGenerator,
    RandomFlatPackGenerator,
)
from jumanji.environments.packing.flat_pack.reward import CellDenseReward, RewardFn
from jumanji.environments.packing.flat_pack.types import Observation, State
from jumanji.environments.packing.flat_pack.utils import compute_grid_dim, rotate_block
from jumanji.environments.packing.flat_pack.viewer import FlatPackViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class FlatPack(Environment[State]):

    """The FlatPack environment with a configurable number of row and column blocks.
    Here the goal of an agent is to completely fill an empty grid by placing all
    available blocks. It can be thought of as a discrete 2D version of the `BinPack`
    environment.

    - observation: `Observation`
        - grid: jax array (int) of shape (num_rows, num_cols) with the
            current state of the grid.
        - blocks: jax array (int) of shape (num_blocks, 3, 3) with the blocks to
            be placed on the grid. Here each block is a 2D array with shape (3, 3).
        - action_mask: jax array (bool) showing where which blocks can be placed on the grid.
            this mask includes all possible rotations and possible placement locations
            for each block on the grid.

    - action: jax array (int32) of shape (4,)
        multi discrete array containing the move to perform
        (block to place, number of rotations, row coordinate, column coordinate).

    - reward: jax array (float) of shape (), could be either:
        - cell dense: the number of non-zero cells in a placed block normalised by the
            total number of cells in a grid. this will be a value in the range [0, 1].
            that is to say that the agent will optimise for the maximum area to fill on
            the grid.
        - block dense: each placed block will receive a reward of 1./num_blocks. this will
            be a value in the range [0, 1]. that is to say that the agent will optimise
            for the maximum number of blocks placed on the grid.
        - sparse: 1 if the grid is completely filled, otherwise 0 at each timestep.

    - episode termination:
        - if all blocks have been placed on the board.
        - if the agent has taken `num_blocks` steps in the environment.

    - state: `State`
        - num_blocks: jax array (int32) of shape () with the
            number of blocks in the environment.
        - blocks: jax array (int32) of shape (num_blocks, 3, 3) with the blocks to
            be placed on the grid. Here each block is a 2D array with shape (3, 3).
        - action_mask: jax array (bool) showing where which blocks can be placed on the grid.
            this mask includes all possible rotations and possible placement locations
            for each block on the grid.
        - placed_blocks: jax array (bool) of shape (num_blocks,) showing which blocks
            have been placed on the grid.
        - grid: jax array (int32) of shape (num_rows, num_cols) with the
            current state of the grid.
        - step_count: jax array (int32) of shape () with the number of steps taken
            in the environment.
        - key: jax array of shape (2,) with the random key used for board
            generation.

    ```python
    from jumanji.environments import FlatPack
    env = FlatPack()
    key = jax.random.PRNGKey(0)
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
        """Initializes the FlatPack environment.

        Args:
            generator: Instance generator for the environment, default to `RandomFlatPackGenerator`
                with a grid of 5 blocks per row and column.
            reward_fn: Reward function for the environment, default to `CellDenseReward`.
            viewer: Viewer for rendering the environment.
        """

        default_generator = RandomFlatPackGenerator(
            num_row_blocks=5,
            num_col_blocks=5,
        )

        self.generator = generator or default_generator
        self.num_row_blocks = self.generator.num_row_blocks
        self.num_col_blocks = self.generator.num_col_blocks
        self.num_blocks = self.num_row_blocks * self.num_col_blocks
        self.num_rows, self.num_cols = (
            compute_grid_dim(self.num_row_blocks),
            compute_grid_dim(self.num_col_blocks),
        )
        self.reward_fn = reward_fn or CellDenseReward()
        self.viewer = viewer or FlatPackViewer(
            "FlatPack", self.num_blocks, render_mode="human"
        )

    def __repr__(self) -> str:
        return (
            f"FlatPack environment with a grid size of ({self.num_rows}x{self.num_cols}) "
            f"with {self.num_row_blocks} row blocks, {self.num_col_blocks} column "
            f"blocks. Each block has dimension (3x3)."
        )

    def reset(
        self,
        key: chex.PRNGKey,
    ) -> Tuple[State, TimeStep[Observation]]:

        """Resets the environment.

        Args:
            key: PRNG key for generating a new instance.

        Returns:
            a tuple of the initial environment state and a time step.
        """

        grid_state = self.generator(key)

        obs = self._observation_from_state(grid_state)
        timestep = restart(observation=obs)

        return grid_state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        """Steps the environment.

        Args:
            state: current state of the environment.
            action: action to take.

        Returns:
            a tuple of the next environment state and a time step.
        """

        # Unpack and use actions
        block_idx, rotation, row_idx, col_idx = action

        chosen_block = state.blocks[block_idx]

        # Rotate chosen block
        chosen_block = rotate_block(chosen_block, rotation)

        grid_block = self._expand_block_to_grid(chosen_block, row_idx, col_idx)

        action_is_legal = state.action_mask[block_idx, rotation, row_idx, col_idx]

        # If the action is legal create a new grid and update the placed blocks array
        new_grid = jax.lax.cond(
            action_is_legal,
            lambda: state.grid + grid_block,
            lambda: state.grid,
        )
        placed_blocks = jax.lax.cond(
            action_is_legal,
            lambda: state.placed_blocks.at[block_idx].set(True),
            lambda: state.placed_blocks,
        )

        new_action_mask = self._make_action_mask(new_grid, state.blocks, placed_blocks)

        next_state = State(
            grid=new_grid,
            blocks=state.blocks,
            action_mask=new_action_mask,
            num_blocks=state.num_blocks,
            key=state.key,
            step_count=state.step_count + 1,
            placed_blocks=placed_blocks,
        )

        done = self._is_done(next_state)
        next_obs = self._observation_from_state(next_state)
        reward = self.reward_fn(state, grid_block, next_state, action_is_legal, done)

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

        Environments will automatically `close()` themselves when
        garbage collected or when the program exits.
        """

        self.viewer.close()

    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec of the environment.

        Returns:
            Spec for each filed in the observation:
            - grid: BoundedArray (int) of shape (num_rows, num_cols).
            - blocks: BoundedArray (int) of shape (num_blocks, 3, 3).
            - action_mask: BoundedArray (bool) of shape
                (num_blocks, 4, num_rows-2, num_cols-2).
        """

        grid = specs.BoundedArray(
            shape=(self.num_rows, self.num_cols),
            minimum=0,
            maximum=self.num_blocks,
            dtype=jnp.int32,
            name="grid",
        )

        blocks = specs.BoundedArray(
            shape=(self.num_blocks, 3, 3),
            minimum=0,
            maximum=self.num_blocks,
            dtype=jnp.int32,
            name="blocks",
        )

        action_mask = specs.BoundedArray(
            shape=(
                self.num_blocks,
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
            grid=grid,
            blocks=blocks,
            action_mask=action_mask,
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Specifications of the action expected by the `FlatPack` environment.

        Returns:
            MultiDiscreteArray (int32) of shape (num_blocks, num_rotations,
            num_rows-2, num_cols-2).
            - num_blocks: int between 0 and num_blocks - 1 (inclusive).
            - num_rotations: int between 0 and 3 (inclusive).
            - max_row_position: int between 0 and num_rows - 3 (inclusive).
            - max_col_position: int between 0 and num_cols - 3 (inclusive).
        """

        max_row_position = self.num_rows - 2
        max_col_position = self.num_cols - 2

        return specs.MultiDiscreteArray(
            num_values=jnp.array(
                [self.num_blocks, 4, max_row_position, max_col_position]
            ),
            name="action",
        )

    def _is_done(self, state: State) -> bool:
        """Checks if the environment is done by checking whether the number of
        steps is equal to the number of blocks.

        Args:
            state: current state of the environment.

        Returns:
            True if the environment is done, False otherwise.
        """

        done: bool = state.step_count >= state.num_blocks

        return done

    def _is_legal_action(
        self,
        action: chex.Numeric,
        grid: chex.Array,
        placed_blocks: chex.Array,
        grid_mask_block: chex.Array,
    ) -> bool:
        """Checks if the action is legal by considering the action mask and the
        current grid. An action is legal if the action mask is True for that action
        and the there is no overlap with blocks already placed.

        Args:
            action: action taken.
            grid: current state of the grid.
            placed_blocks: array indicating which blocks have been placed.
            grid_mask_block: grid with ones where current block should be placed.

        Returns:
            True if the action is legal, False otherwise.
        """

        block_idx, _, _, _ = action

        placed_mask = (grid > 0.0) + grid_mask_block

        legal: bool = (~placed_blocks[block_idx]) & (jnp.max(placed_mask) <= 1)

        return legal

    def _get_ones_like_expanded_block(self, grid_block: chex.Array) -> chex.Array:
        """Makes a grid of zeroes with ones where the block is placed."""

        return (grid_block != 0).astype(jnp.int32)

    def _expand_block_to_grid(
        self,
        block: chex.Array,
        row_coord: chex.Numeric,
        col_coord: chex.Numeric,
    ) -> chex.Array:
        """Places a block on a grid of zeroes with the same size as the grid.

        Args:
            block: block to place on the grid.
            row_coord: row coordinate on the grid where the top left corner
                of the block will be placed.
            col_coord: column coordinate on the grid where the top left corner
                of the block will be placed.

        Returns:
            Grid of zeroes with values where the block is placed.
        """

        # Make an empty grid for placing the block on.
        grid_with_block = jnp.zeros((self.num_rows, self.num_cols), dtype=jnp.int32)
        place_location = (row_coord, col_coord)

        grid_with_block = jax.lax.dynamic_update_slice(
            grid_with_block, block, place_location
        )

        return grid_with_block

    def _observation_from_state(self, state: State) -> Observation:
        """Creates an observation from a state.

        Args:
            state: State to create an observation from.

        Returns:
            An observation.
        """

        return Observation(
            grid=state.grid,
            action_mask=state.action_mask,
            blocks=state.blocks,
        )

    def _expand_all_blocks_to_grids(
        self,
        blocks: chex.Array,
        block_idxs: chex.Array,
        rotations: chex.Array,
        row_coords: chex.Array,
        col_coords: chex.Array,
    ) -> chex.Array:
        """Takes multiple blocks and their corresponding rotations and positions,
            and generates a grid for each block.

        Args:
            blocks: array of possible blocks.
            block_idxs: array of indices of the blocks to place.
            rotations: array of all possible rotations for each block.
            row_coords: array of row coordinates.
            col_coords: array of column coordinates.
        """

        batch_expand_block_to_board = jax.vmap(self._expand_block_to_grid)

        all_possible_blocks = blocks[block_idxs]
        rotated_blocks = jax.vmap(rotate_block)(all_possible_blocks, rotations)
        grids = batch_expand_block_to_board(rotated_blocks, row_coords, col_coords)

        batch_get_ones_like_expanded_block = jax.vmap(
            self._get_ones_like_expanded_block, in_axes=(0)
        )
        grids = batch_get_ones_like_expanded_block(grids)
        return grids

    def _make_action_mask(
        self, grid: chex.Array, blocks: chex.Array, placed_blocks: chex.Array
    ) -> chex.Array:
        """Create a mask of possible actions based on the current state of the grid.

        Args:
            grid: current state of the grid.
            blocks: array of all blocks.
            placed_blocks: array of blocks that have already been placed.
        """

        num_blocks, num_rotations, num_placement_rows, num_placement_cols = (
            self.num_blocks,
            4,
            self.num_rows - 2,
            self.num_cols - 2,
        )

        blocks_grid, rotations_grid, rows_grid, cols_grid = jnp.meshgrid(
            jnp.arange(num_blocks),
            jnp.arange(num_rotations),
            jnp.arange(num_placement_rows),
            jnp.arange(num_placement_cols),
            indexing="ij",
        )

        grid_mask_pieces = self._expand_all_blocks_to_grids(
            blocks,
            blocks_grid.flatten(),
            rotations_grid.flatten(),
            rows_grid.flatten(),
            cols_grid.flatten(),
        )

        batch_is_legal_action = jax.vmap(
            self._is_legal_action, in_axes=(0, None, None, 0)
        )

        all_actions = jnp.stack(
            (blocks_grid, rotations_grid, rows_grid, cols_grid), axis=-1
        ).reshape(-1, 4)

        legal_actions = batch_is_legal_action(
            all_actions,
            grid,
            placed_blocks,
            grid_mask_pieces,
        )

        legal_actions = legal_actions.reshape(
            num_blocks, num_rotations, num_placement_rows, num_placement_cols
        )

        # Now set all current placed blocks to false in the mask.
        placed_blocks_array = placed_blocks.reshape((self.num_blocks, 1, 1, 1))
        placed_blocks_mask = jnp.tile(
            placed_blocks_array,
            (1, num_rotations, num_placement_rows, num_placement_cols),
        )
        legal_actions = jnp.where(placed_blocks_mask, False, legal_actions)

        return legal_actions
