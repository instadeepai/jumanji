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
        """Return new updated carry. This method should not be called from within a jax.lax.cond."""
        row = self.row
        row = row.at[self.target_idx].set(update.target)
        row = row.at[self.origin_idx].set(update.origin)
        return self._replace(
            row=row,
            reward=self.reward + update.additional_reward,
            target_idx=update.target_idx,
            origin_idx=update.origin_idx,
        )


def no_op(carry: MoveCarry) -> MoveUpdate:
    """Returns a move update equivalent to performing a no op."""
    target_idx = jax.lax.cond(
        carry.origin == 0,
        lambda: carry.target_idx,
        lambda: carry.target_idx + 1,
    )
    origin_idx = jax.lax.cond(
        (carry.origin == 0) | (target_idx == carry.origin_idx),
        lambda: carry.origin_idx + 1,
        lambda: carry.origin_idx,
    )
    return MoveUpdate(
        target=carry.target,
        origin=carry.origin,
        additional_reward=0.0,
        target_idx=target_idx,
        origin_idx=origin_idx,
    )


def shift(carry: MoveCarry) -> MoveUpdate:
    """Returns a move update equivalent to shifting origin to target."""
    return MoveUpdate(
        target=carry.origin,
        origin=0,
        additional_reward=0.0,
        target_idx=carry.target_idx,
        origin_idx=carry.origin_idx + 1,
    )


def merge(carry: MoveCarry) -> MoveUpdate:
    """Returns a move update equivalent to merging origin with target."""
    return MoveUpdate(
        target=carry.target + 1,
        origin=0,
        target_idx=carry.target_idx + 1,
        origin_idx=carry.origin_idx + 1,
        additional_reward=2.0 ** (carry.target + 1),
    )


def move_left_row_cond(carry: MoveCarry) -> chex.Numeric:
    """Terminates loop when origin reaches end of row."""
    return carry.origin_idx < carry.row.shape[0]


def move_left_row_body(carry: MoveCarry) -> MoveCarry:
    """Returns new carry after performing appropiate op."""
    can_shift = (carry.origin != 0) & (carry.target == 0)
    can_merge = (carry.origin != 0) & (carry.target == carry.origin)
    move_type = can_shift.astype(int) + 2 * can_merge.astype(int)
    update = jax.lax.switch(move_type, [no_op, shift, merge], carry)
    return carry.update(update)


def move_left_row(row: chex.Array) -> Tuple[chex.Array, float]:
    """Move the elements in the row left."""
    carry = MoveCarry(
        row=row,
        reward=0.0,
        target_idx=0,
        origin_idx=1,
    )
    row, reward = jax.lax.while_loop(move_left_row_cond, move_left_row_body, carry)[:2]
    return row, reward


def move_left(board: Board) -> Tuple[Board, float]:
    """Move left."""
    board, reward = jax.vmap(move_left_row)(board)
    return board, reward.sum()


def transform_board(board: Board, action: int) -> Board:
    """Transform board."""
    return jax.lax.switch(
        action,
        [
            lambda board: jnp.transpose(board),
            lambda board: jnp.flip(board, 1),
            lambda board: jnp.flip(jnp.transpose(board)),
            lambda board: board,
        ],
        board,
    )


def move(board: Board, action: int) -> Tuple[Board, float]:
    """Move."""
    board = transform_board(board, action)
    board, reward = move_left(board)
    board = transform_board(board, action)
    return board, reward


def move_up(board: Board) -> Tuple[Board, float]:
    """Move up."""
    return move(board=board, action=0)


def move_right(board: Board) -> Tuple[Board, float]:
    """Move right."""
    return move(board=board, action=1)


def move_down(board: Board) -> Tuple[Board, float]:
    """Move down."""
    return move(board=board, action=2)
