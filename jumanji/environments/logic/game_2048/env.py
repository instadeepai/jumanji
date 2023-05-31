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
import matplotlib.animation as animation
from numpy.typing import NDArray

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.logic.game_2048.types import Board, Observation, State
from jumanji.environments.logic.game_2048.utils import (
    move_down,
    move_left,
    move_right,
    move_up,
)
from jumanji.environments.logic.game_2048.viewer import Game2048Viewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class Game2048(Environment[State]):
    """Environment for the game 2048. The game consists of a board of size board_size x board_size
    (4x4 by default) in which the player can take actions to move the tiles on the board up, down,
    left, or right. The goal of the game is to combine tiles with the same number to create a tile
    with twice the value, until the player at least creates a tile with the value 2048 to consider
    it a win.

    - observation: `Observation`
        - board: jax array (int32) of shape (board_size, board_size)
            the current state of the board. An empty tile is represented by zero whereas a non-empty
            tile is an exponent of 2, e.g. 1, 2, 3, 4, ... (corresponding to 2, 4, 8, 16, ...).
        - action_mask: jax array (bool) of shape (4,)
            indicates which actions are valid in the current state of the environment.

    - action: jax array (int32) of shape (). Is in [0, 1, 2, 3] representing the actions up, right,
        down, and left, respectively.

    - reward: jax array (float) of shape (). The reward is 0 except when the player combines tiles
        to create a new tile with twice the value. In this case, the reward is the value of the new
        tile.

    - episode termination:
        - if no more valid moves exist (this can happen when the board is full).

    - state: `State`
        - board: same as observation.
        - step_count: jax array (int32) of shape (),
            the number of time steps in the episode so far.
        - action_mask: same as observation.
        - score: jax array (int32) of shape (),
            the sum of all tile values on the board.
        - key: jax array (uint32) of shape (2,)
            random key used to generate random numbers at each step and for auto-reset.

    ```python
    from jumanji.environments import Game2048
    env = Game2048()
    key = jax.random.key(0)
    state, timestep = jax.jit(env.reset)(key)
    env.render(state)
    action = env.action_spec().generate_value()
    state, timestep = jax.jit(env.step)(state, action)
    env.render(state)
    ```
    """

    def __init__(
        self, board_size: int = 4, viewer: Optional[Viewer[State]] = None
    ) -> None:
        """Initialize the 2048 game.

        Args:
            board_size: size of the board. Defaults to 4.
            viewer: `Viewer` used for rendering. Defaults to `Game2048Viewer`.
        """
        self.board_size = board_size

        # Create viewer used for rendering
        self._viewer = viewer or Game2048Viewer("2048", board_size)

    def __repr__(self) -> str:
        """String representation of the environment.

        Returns:
            str: the string representation of the environment.
        """
        return f"2048 Game(board_size={self.board_size})"

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specifications of the observation of the `Game2048` environment.

        Returns:
            Spec containing all the specifications for all the `Observation` fields:
             - board: Array (jnp.int32) of shape (board_size, board_size).
             - action_mask: BoundedArray (bool) of shape (4,).
        """
        return specs.Spec(
            Observation,
            "ObservationSpec",
            board=specs.Array(
                shape=(self.board_size, self.board_size),
                dtype=jnp.int32,
                name="board",
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
        """Returns the action spec.

        4 actions: [0, 1, 2, 3] -> [Up, Right, Down, Left].

        Returns:
            action_spec: `DiscreteArray` spec object.
        """
        return specs.DiscreteArray(4, name="action")

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment.

        Args:
            key: random number generator key.

        Returns:
            state: the new state of the environment.
            timestep: the first timestep returned by the environment.
        """

        key, board_key = jax.random.split(key)
        board = self._generate_board(board_key)
        action_mask = self._get_action_mask(board)

        obs = Observation(board=board, action_mask=action_mask)

        state = State(
            board=board,
            step_count=jnp.array(0, jnp.int32),
            action_mask=action_mask,
            key=key,
            score=jnp.array(0, float),
        )

        highest_tile = 2 ** jnp.max(board)
        timestep = restart(observation=obs, extras={"highest_tile": highest_tile})

        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        """Updates the environment state after the agent takes an action.

        Args:
            state: the current state of the environment.
            action: the action taken by the agent.

        Returns:
            state: the new state of the environment.
            timestep: the next timestep.
        """
        # Take the action in the environment: Up, Right, Down, Left.
        updated_board, additional_reward = jax.lax.switch(
            action,
            [move_up, move_right, move_down, move_left],
            state.board,
        )

        # Generate new key.
        random_cell_key, new_state_key = jax.random.split(state.key)

        # Update the state of the board by adding a new random cell.
        updated_board = jax.lax.cond(
            state.action_mask[action],
            self._add_random_cell,
            lambda board, key: board,
            updated_board,
            random_cell_key,
        )

        # Generate action mask to keep in the state for the next step and
        # to provide to the agent in the observation.
        action_mask = self._get_action_mask(board=updated_board)

        # Build the state.
        state = State(
            board=updated_board,
            action_mask=action_mask,
            step_count=state.step_count + 1,
            key=new_state_key,
            score=state.score + additional_reward.astype(float),
        )

        # Generate the observation from the environment state.
        observation = Observation(
            board=updated_board,
            action_mask=action_mask,
        )

        # Check if the episode terminates (i.e. there are no legal actions).
        done = ~jnp.any(action_mask)

        # Return either a MID or a LAST timestep depending on done.
        highest_tile = 2 ** jnp.max(updated_board)
        extras = {"highest_tile": highest_tile}
        timestep = jax.lax.cond(
            done,
            lambda: termination(
                reward=additional_reward,
                observation=observation,
                extras=extras,
            ),
            lambda: transition(
                reward=additional_reward,
                observation=observation,
                extras=extras,
            ),
        )

        return state, timestep

    def _generate_board(self, key: chex.PRNGKey) -> Board:
        """Generates an initial board for the environment.

        The method generates an empty board with the specified size and fills a random cell with
        a value of 1 or 2 representing the exponent of 2.

        Args:
            key: random number generator key.

        Returns:
            board: initial board for the environment.
        """
        # Create empty board
        board = jnp.zeros((self.board_size, self.board_size), dtype=jnp.int32)

        # Fill one random cell with a value of 1 or 2
        board = self._add_random_cell(board, key)

        return board

    def _add_random_cell(self, board: Board, key: chex.PRNGKey) -> Board:
        """Adds a new random cell to the board.

        This method selects an empty position in the board and assigns it a value
        of 1 or 2 representing the exponent of 2.

        Args:
            board: current board of the environment.
            key: random number generator key.

        Returns:
            board: updated board with the new random cell added.
        """
        key, subkey = jax.random.split(key)

        # Select position of the new random cell
        empty_flatten_board = jnp.ravel(board == 0)
        tile_idx = jax.random.choice(
            key, jnp.arange(len(empty_flatten_board)), p=empty_flatten_board
        )
        # Convert the selected tile's location in the flattened array to its position on the board.
        position = jnp.divmod(tile_idx, self.board_size)

        # Choose the value of the new cell: 1 with probability 90% or 2 with probability of 10%
        cell_value = jax.random.choice(
            subkey, jnp.array([1, 2]), p=jnp.array([0.9, 0.1])
        )
        board = board.at[position].set(cell_value)

        return board

    def _get_action_mask(self, board: Board) -> chex.Array:
        """Generates a binary mask indicating which actions are valid.

        If the movement in that direction leaves the board unchanged, the action is
        considered illegal.

        Args:
            board: current board of the environment.

        Returns:
            action_mask: action mask for the current state of the environment.
        """
        action_mask = jnp.array(
            [
                jnp.any(move_up(board, final_shift=False)[0] != board),
                jnp.any(move_right(board, final_shift=False)[0] != board),
                jnp.any(move_down(board, final_shift=False)[0] != board),
                jnp.any(move_left(board, final_shift=False)[0] != board),
            ],
        )
        return action_mask

    def render(self, state: State) -> Optional[NDArray]:
        """Renders the current state of the game board.

        Args:
            state: is the current game state to be rendered.
        """
        return self._viewer.render(state=state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> animation.FuncAnimation:
        """Creates an animated gif of the 2048 game board based on the sequence of game states.

        Args:
            states: is a list of `State` objects representing the sequence of game states.
            interval: the delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
            will not be stored.

        Returns:
            animation.FuncAnimation: the animation object that was created.
        """
        return self._viewer.animate(
            states=states, interval=interval, save_path=save_path
        )

    def close(self) -> None:
        """Perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        self._viewer.close()
