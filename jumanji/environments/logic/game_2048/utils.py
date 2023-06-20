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

from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.logic.game_2048.types import Board


def transform_board(board: Board, action: int) -> Board:
    """Transform board so that move_left is analagous to move_action. Also, transform back."""
    return jax.lax.switch(
        action,
        [
            lambda: jnp.transpose(board),
            lambda: jnp.flip(board, 1),
            lambda: jnp.flip(jnp.transpose(board)),
            lambda: board,
        ],
    )


class CanMoveCarry(NamedTuple):
    """Carry value for while loop in can_move_left_row."""

    can_move: bool
    row: chex.Array
    target_idx: int
    origin_idx: int

    @property
    def target(self) -> chex.Numeric:
        """Tile at target index of row."""
        return self.row[self.target_idx]

    @property
    def origin(self) -> chex.Numeric:
        """Tile at origin index of row."""
        return self.row[self.origin_idx]


def can_move_left_row_cond(carry: CanMoveCarry) -> chex.Numeric:
    """Terminate loop when valid move is found or origin reaches end of row."""
    return ~carry.can_move & (carry.origin_idx < carry.row.shape[0])


def can_move_left_row_body(carry: CanMoveCarry) -> CanMoveCarry:
    """Check if the current tiles can move and increment the indices."""
    # Check if tiles can move
    can_move = (carry.origin != 0) & (
        (carry.target == 0) | (carry.target == carry.origin)
    )

    # Increment indices as if performed a no op
    # If not performing no op, loop will be terminated anyways
    target_idx = carry.target_idx + (carry.origin != 0)
    origin_idx = jax.lax.select(
        (carry.origin == 0) | (target_idx == carry.origin_idx),
        carry.origin_idx + 1,
        carry.origin_idx,
    )

    # Return updated carry
    return carry._replace(
        can_move=can_move, target_idx=target_idx, origin_idx=origin_idx
    )


def can_move_left_row(row: chex.Array) -> bool:
    """Check if row can move left."""
    carry = CanMoveCarry(can_move=False, row=row, target_idx=0, origin_idx=1)
    can_move: bool = jax.lax.while_loop(
        can_move_left_row_cond, can_move_left_row_body, carry
    )[0]
    return can_move


def can_move_left(board: Board) -> bool:
    """Check if board can move left."""
    can_move: bool = jax.vmap(can_move_left_row)(board).any()
    return can_move


def can_move(board: Board, action: int) -> bool:
    """Check if board can move with action."""
    return can_move_left(transform_board(board, action))


def can_move_up(board: Board) -> bool:
    """Check if board can move up."""
    return can_move(board, 0)


def can_move_right(board: Board) -> bool:
    """Check if board can move right."""
    return can_move(board, 1)


def can_move_down(board: Board) -> bool:
    """Check if board can move down."""
    return can_move(board, 2)


class MoveUpdate(NamedTuple):
    """Update to move carry."""

    target: chex.Numeric
    origin: chex.Numeric
    additional_reward: float
    target_idx: int
    origin_idx: int


class MoveCarry(NamedTuple):
    """Carry value for while loop in move_left_row."""

    row: chex.Array
    reward: float
    target_idx: int
    origin_idx: int

    @property
    def target(self) -> chex.Numeric:
        """Tile at target index of row."""
        return self.row[self.target_idx]

    @property
    def origin(self) -> chex.Numeric:
        """Tile at origin index of row."""
        return self.row[self.origin_idx]

    def update(self, update: MoveUpdate) -> "MoveCarry":
        """Return new updated carry. This method will cause row to be copied when called within a
        jax conditional primative such as `jax.lax.cond` or `jax.lax.switch`.
        """
        # Update row
        row = self.row
        row = row.at[self.target_idx].set(update.target)
        row = row.at[self.origin_idx].set(update.origin)

        # Return updated carry
        return self._replace(
            row=row,
            reward=self.reward + update.additional_reward,
            target_idx=update.target_idx,
            origin_idx=update.origin_idx,
        )


def no_op(carry: MoveCarry) -> MoveUpdate:
    """Return a move update equivalent to performing a no op."""
    target_idx = carry.target_idx + (carry.origin != 0)
    origin_idx = jax.lax.select(
        (carry.origin == 0) | (target_idx == carry.origin_idx),
        carry.origin_idx + 1,
        carry.origin_idx,
    )
    return MoveUpdate(
        target=carry.target,
        origin=carry.origin,
        additional_reward=0.0,
        target_idx=target_idx,
        origin_idx=origin_idx,
    )


def shift(carry: MoveCarry) -> MoveUpdate:
    """Return a move update equivalent to shifting origin to target."""
    return MoveUpdate(
        target=carry.origin,
        origin=0,
        additional_reward=0.0,
        target_idx=carry.target_idx,
        origin_idx=carry.origin_idx + 1,
    )


def merge(carry: MoveCarry) -> MoveUpdate:
    """Return a move update equivalent to merging origin with target."""
    return MoveUpdate(
        target=carry.target + 1,
        origin=0,
        additional_reward=2.0 ** (carry.target + 1),
        target_idx=carry.target_idx + 1,
        origin_idx=carry.origin_idx + 1,
    )


def move_left_row_cond(carry: MoveCarry) -> chex.Numeric:
    """Terminate loop when origin reaches end of row."""
    return carry.origin_idx < carry.row.shape[0]


def move_left_row_body(carry: MoveCarry) -> MoveCarry:
    """Move the current tiles and increment the indices."""
    # Determine move type
    can_shift = (carry.origin != 0) & (carry.target == 0)
    can_merge = (carry.origin != 0) & (carry.target == carry.origin)
    move_type = can_shift.astype(int) + 2 * can_merge.astype(int)

    # Get update based on move type
    update = jax.lax.switch(move_type, [no_op, shift, merge], carry)

    # Return updated carry
    return carry.update(update)


def move_left_row(row: chex.Array) -> Tuple[chex.Array, float]:
    """Move the row left."""
    carry = MoveCarry(row=row, reward=0.0, target_idx=0, origin_idx=1)
    row, reward, *_ = jax.lax.while_loop(move_left_row_cond, move_left_row_body, carry)
    return row, reward


def move_left(board: Board) -> Tuple[Board, float]:
    """Move the board left."""
    board, reward = jax.vmap(move_left_row)(board)
    return board, reward.sum()


def move(board: Board, action: int) -> Tuple[Board, float]:
    """Move the board with action."""
    board = transform_board(board, action)
    board, reward = move_left(board)
    board = transform_board(board, action)
    return board, reward


def move_up(board: Board) -> Tuple[Board, float]:
    """Move the board up."""
    return move(board, 0)


def move_right(board: Board) -> Tuple[Board, float]:
    """Move the board right."""
    return move(board, 1)


def move_down(board: Board) -> Tuple[Board, float]:
    """Move the board down."""
    return move(board, 2)
