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

from typing import Dict, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib.animation as animation
from numpy.typing import NDArray

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.logic.solitaire.types import Board, Observation, State
from jumanji.environments.logic.solitaire.utils import (
    all_possible_moves,
    move,
    playing_board,
)
from jumanji.environments.logic.solitaire.viewer import SolitaireViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class Solitaire(Environment[State]):
    """Environment for peg solitaire. https://en.wikipedia.org/wiki/Peg_solitaire

    - observation: `Observation`
        - board: jax array (bool) of shape (board_size, board_size)
            the current state of the board. A peg is represented by 1 and hole is represented by 0.
        - action_mask: jax array (bool) of shape (board_size, board_size, 4,)
            indicates which actions are valid in the current state of the environment.

    - action: jax array (int32) of shape (3, ). The first two positions represent the board position
        of the peg location and the third poition is the direction to move the peg.
    - reward: jax array (float) of shape (). The reward is 1 at each move.

    - episode termination:
        - if no more valid moves exist (this can happen when the game is solved).

    - state: `State`
        - board: same as observation.
        - step_count: jax array (int32) of shape (),
            the number of time steps in the episode so far.
        - action_mask: same as observation.
        - remaining: jax array (int32) of shape (),
            the number of pegs remaining on the board
        - key: jax array (uint32) of shape (2,)
            random key used to generate random numbers at each step and for auto-reset.

    ```python
    from jumanji.environments import Solitaire
    env = Solitaire()
    key = jax.random.key(0)
    state, timestep = jax.jit(env.reset)(key)
    env.render(state)
    action = env.action_spec().generate_value()
    state, timestep = jax.jit(env.step)(state, action)
    env.render(state)
    ```
    """

    def __init__(
        self, board_size: int = 7, viewer: Optional[Viewer[State]] = None
    ) -> None:
        """Initialize the solitaire board.

        Args:
            board_size: size of the board (must be odd). Defaults to 7.
            viewer: `Viewer` used for rendering. Defaults to `SolitaireViewer`.
        """
        if board_size % 2 == 0:
            raise ValueError("`board_size` must be odd.")
        self.board_size = board_size
        self.mid_size = board_size // 2

        # Create viewer used for rendering
        self._viewer = viewer or SolitaireViewer("solitaire", board_size)

    def __repr__(self) -> str:
        """String representation of the environment.

        Returns:
            str: the string representation of the environment.
        """
        return f"Solitaire(board_size={self.board_size})"

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specifications of the observation of the `Solitaire` environment.

        Returns:
            Spec containing all the specifications for all the `Observation` fields:
             - board: Array (jnp.int32) of shape (board_size, board_size).
             - action_mask: BoundedArray (bool) of shape (4,).
        """
        return specs.Spec(
            Observation,
            "ObservationSpec",
            board=specs.BoundedArray(
                shape=(self.board_size, self.board_size),
                minimum=False,
                maximum=True,
                dtype=bool,
                name="board",
            ),
            action_mask=specs.BoundedArray(
                shape=(
                    self.board_size,
                    self.board_size,
                    4,
                ),
                dtype=bool,
                minimum=False,
                maximum=True,
                name="action_mask",
            ),
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Specifications of the action expected by the `BinPack` environment.

        Returns:
            MultiDiscreteArray (int32) of shape (board_size, board_size, 4).
            - row of the peg to move.
            - column of the peg to move.
            - direction of the move: 0 for up, 1 for right, 2 for down, 3 for left.
        """
        num_actions = jnp.array((self.board_size, self.board_size, 4), jnp.int32)
        return specs.MultiDiscreteArray(num_actions, name="action")

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment.

        Args:
            key: random number generator key.
                Note: the key is not used in this environment but is there
                for consistency and future development.

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
            remaining=jnp.sum(board),
        )

        timestep = restart(observation=obs, extras={})

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
        action_is_valid = state.action_mask[tuple(action)]  # type: ignore

        # Take the action in the environment.
        updated_board, reward = jax.lax.cond(
            action_is_valid,
            lambda s: move(state.board, action),
            lambda s: (state.board, 0.0),
            state,
        )
        stepped = ~jnp.array_equal(state.board, updated_board)

        # Generate action mask to keep in the state for the next step and
        # to provide to the agent in the observation.
        action_mask = jax.lax.cond(
            stepped,
            lambda s: self._get_action_mask(updated_board),
            lambda s: s.action_mask,
            state,
        )

        # Build the state.
        state = State(
            board=updated_board,
            action_mask=action_mask,
            step_count=state.step_count + 1,
            key=state.key,
            remaining=state.remaining - stepped,
        )

        # Generate the observation from the environment state.
        observation = Observation(
            board=updated_board,
            action_mask=action_mask,
        )

        # Check if the episode terminates (i.e. there are no legal actions).
        done = ~jnp.any(action_mask)

        # Return either a MID or a LAST timestep depending on done.
        extras: Dict = {}
        timestep = jax.lax.cond(
            done,
            lambda: termination(
                reward=reward,
                observation=observation,
                extras=extras,
            ),
            lambda: transition(
                reward=reward,
                observation=observation,
                extras=extras,
            ),
        )

        return state, timestep

    def _generate_full_board(self, key: chex.PRNGKey) -> Board:
        """Generates board with all the pegs in."""
        return playing_board(self.board_size)

    def _generate_board(self, key: chex.PRNGKey) -> Board:
        """Generates an initial board for the environment.


        The method generates a full board and then removes the central tile.
        Args:
            key: random number generator key.

        Returns:
            board: initial board for the environment.
        """
        # Create full board.
        board = self._generate_full_board(key)

        # Remove the centre peg.
        board = board.at[self.mid_size, self.mid_size].set(False)

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
        return all_possible_moves(board)

    def render(self, state: State) -> Optional[NDArray]:
        """Renders the current state of the board.

        Args:
            state: is the current state to be rendered.
        """
        return self._viewer.render(state=state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 500,
        save_path: Optional[str] = None,
    ) -> animation.FuncAnimation:
        """Creates an animated gif of solitaire based on the sequence of states.

        Args:
            states: is a list of `State` objects representing the sequence of states.
            interval: the delay between frames in milliseconds, default to 500.
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
