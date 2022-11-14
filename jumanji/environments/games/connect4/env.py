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

from chex import Array, PRNGKey
from jax import lax
from jax import numpy as jnp

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.games.connect4.constants import BOARD_HEIGHT, BOARD_WIDTH
from jumanji.environments.games.connect4.specs import ObservationSpec
from jumanji.environments.games.connect4.types import Observation, State
from jumanji.environments.games.connect4.utils import (
    board_full,
    get_action_mask,
    get_highest_row,
    is_winning,
    update_board,
)
from jumanji.types import Action, TimeStep, restart, termination, transition


class Connect4(Environment[State]):
    """A JAX implementation of the 'Connect 4' game.

    - observation: Observation
        - board: jax array (int8) of shape (6, 7):
            each cell contains either:
            - 1 if it contains a token by the current player,
            - 0 if it is free,
            - (-1) if it contains a token by the other player.
        - action_mask: jax array (bool)
            valid columns (actions) are identified with `True`, invalid ones with `False`.
        - current_player: jnp.int8, id of the current player {0, 1}.

    - action: Array containing the column to insert the token into {0, 1, 2, 3, 4, 5, 6}

    - reward: jax array (float) of shape (2,).
        - 1 for the winning player, 0 for a draw and -1 for the losing player.

    - episode termination:
        - if the board is full, the game ends on a draw.
        - if a player connects 4 tokens (horizontally, vertically or diagonally), they win
        and the game ends.
        - if a player plays an invalid move, this player loses and the game ends.

    - state: State
        - current_player: jnp.int8, id of the current player {0, 1}.
        - board: jax array (int8) of shape (6, 7):
            each cell contains either:
            - 1 if it contains a token placed by the current player,
            - 0 if it is free,
            - (-1) if it contains a token placed by the other player.

    """

    n_players: int = 2

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment.

        Args:
            key: not used.

        Returns:
            state: State object corresponding to the new state of the environment,
            timestep: TimeStep object corresponding to the first timestep returned by the
            environment. Its `observation` attribute contains a field for the current player id.
        """
        del key
        board = jnp.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=jnp.int8)
        action_mask = jnp.ones((BOARD_WIDTH,), dtype=jnp.int8)

        obs = Observation(
            board=board, action_mask=action_mask, current_player=jnp.int8(0)
        )

        timestep = restart(observation=obs, shape=(self.n_players,))

        state = State(current_player=jnp.int8(0), board=board)

        return state, timestep

    def step(self, state: State, action: Action) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the column to insert the token into {0, 1, 2, 3, 4, 5, 6}

        Returns:
            state: State object corresponding to the next state of the environment,
            timestep: TimeStep object corresponding to the timestep returned by the environment.
                Its `observation` attribute contains a field for the current player id.
        """
        board = state.board

        # getting the height of the column
        highest_row = get_highest_row(board[:, action])

        # checking the validity of the move
        invalid = jnp.any(highest_row == 0)

        # applying action
        new_board = lax.cond(
            invalid, lambda x: x, lambda b: update_board(b, highest_row, action), board
        )

        # computing winning condition
        winning = is_winning(new_board)

        # computing terminal condition
        done = invalid | winning | board_full(new_board)

        # computing action mask
        action_mask = get_action_mask(new_board)

        # switching player
        next_player = jnp.int8((state.current_player + 1) % self.n_players)

        # computing reward
        reward_value = compute_reward(invalid, winning)

        reward = jnp.zeros((self.n_players,))
        reward = reward.at[state.current_player].set(reward_value)

        # opponent gets the opposite reward of the current player
        reward = reward.at[next_player].set(-reward_value)

        # creating next state
        next_state = State(current_player=next_player, board=new_board)

        obs = Observation(
            board=new_board,
            action_mask=action_mask,
            current_player=next_player,
        )

        timestep = lax.cond(
            done,
            lambda _: termination(
                reward=reward,
                observation=obs,
                shape=(self.n_players,),
            ),
            lambda _: transition(
                reward=reward,
                observation=obs,
                shape=(self.n_players,),
            ),
            operand=None,
        )

        return next_state, timestep

    def observation_spec(self) -> ObservationSpec:
        """Returns the observation spec containing the board and action_mask arrays.

        Returns:
            observation_spec: ObservationSpec tree of board and action_mask spec.
        """
        return ObservationSpec(
            board_obs=specs.Array(shape=(6, 7), dtype=jnp.int8, name="board"),
            action_mask=specs.BoundedArray(
                shape=(7,),
                dtype=bool,
                minimum=0,
                maximum=1,
                name="invalid_mask",
            ),
            current_player=specs.DiscreteArray(
                num_values=self.n_players, dtype=jnp.int8, name="current_player"
            ),
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec. 7 actions: [0,1,2,3,4,5,6] -> one per column.

        Returns:
            action_spec: specs.DiscreteArray object
        """
        return specs.DiscreteArray(7, name="action")

    def render(self, state: State) -> str:
        """Renders a given state.

        Args:
            state: State object corresponding to the new state of the environment.

        Returns:
            human-readable string displaying the current state of the game.

        """
        message = f"Current player: {state.current_player}\n"
        message += f"Board: \n {str(state.board)}"
        return message

    def close(self) -> None:
        """Perform any necessary cleanup. Since the rendering returns a string only, nothing needs
        to be done to close, therefore, this method does not do anything.
        """


def compute_reward(invalid: Array, winning: Array) -> Array:
    """Computes the reward based on the validity of the move of the current player and whether it
    was a winning move or not.

    Reward is as follows:
        - if the move was a winning move then the player receives 1,
        - if it was an invalid move, the player receives -1.
        - otherwise, the player receives 0.

    Args:
        invalid: True means the move was not valid,
        winning: True means the move was a winning move.

    Returns:
        The reward
    """
    reward = lax.cond(winning, lambda _: 1, lambda _: 0, operand=None)
    return lax.cond(invalid, lambda _: -1, lambda r: r, operand=reward)
