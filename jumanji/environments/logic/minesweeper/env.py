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

from chex import PRNGKey
from jax import lax
from jax import numpy as jnp

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.logic.minesweeper.constants import (
    DEFAULT_BOARD_HEIGHT,
    DEFAULT_BOARD_WIDTH,
    DEFAULT_NUM_MINES,
    UNEXPLORED_ID,
)
from jumanji.environments.logic.minesweeper.done_functions import (
    DefaultDoneFunction,
    DoneFunction,
)
from jumanji.environments.logic.minesweeper.reward_functions import (
    DefaultRewardFunction,
    RewardFunction,
)
from jumanji.environments.logic.minesweeper.specs import ObservationSpec
from jumanji.environments.logic.minesweeper.types import Observation, State
from jumanji.environments.logic.minesweeper.utils import (
    count_adjacent_mines,
    create_flat_mine_locations,
)
from jumanji.types import Action, TimeStep, restart, termination, transition


def state_to_observation(state: State, num_mines: int) -> Observation:
    return Observation(
        board=state.board,
        action_mask=jnp.equal(state.board, UNEXPLORED_ID),
        num_mines=jnp.int32(num_mines),
        step_count=state.step_count,
    )


class Minesweeper(Environment[State]):
    """A JAX implementation of the minesweeper game.

    - observation: Observation
        - board: jax array (int32) of shape (board_height, board_width):
            each cell contains -1 if not yet explored, or otherwise the number of mines in
            the 8 adjacent squares.
        - action_mask: jax array (bool) of same shape as board:
            indicates which actions are valid (not yet explored squares).
        - num_mines: the number of mines to find, which can be read from the env
        - step_count: jax array (int32):
            specifies how many timesteps have elapsed since environment reset

    - action:
        multi discrete array containing the square to explore (height and width)

    - reward: jax array (float32):
        Configurable function of state and action. By default:
            1 for every timestep where a valid action is chosen that doesn't reveal a mine
            0 for revealing a mine or selecting an already revealed square
                (and terminate the episode)

    - episode termination:
        Configurable function of state, next_state, and action. By default:
            Stop the episode if a mine is explored, an invalid action is selected
            (exploring an already explored square), or the board is solved.

    - state: State
        - board: as in observation.
        - step_count: as in observation.
        - flat_mine_locations: jax array (int32) of shape (board_height * board_width):
            indicates the (flat) locations of all of the mines on the board.
            Will be of length num_mines.
        - key: jax array (int32) of shape (2) used for seeding the sampling of mine placement
            on reset.
    """

    def __init__(
        self,
        board_height: int = DEFAULT_BOARD_HEIGHT,
        board_width: int = DEFAULT_BOARD_WIDTH,
        reward_function_type: str = "default",
        done_function_type: str = "default",
        num_mines: int = DEFAULT_NUM_MINES,
    ):
        if board_height <= 1 or board_width <= 1:
            raise ValueError(
                f"Should make a board of height and width greater than 1, "
                f"got height={board_height}, width={board_width}"
            )
        if num_mines < 0 or num_mines >= board_height * board_width:
            raise ValueError(
                f"Number of mines should be constrained between 0 and the size of the board, "
                f"got {num_mines}"
            )
        self.board_height = board_height
        self.board_width = board_width
        self.num_mines = num_mines
        self.reward_function = self.create_reward_function(
            reward_function_type=reward_function_type
        )
        self.done_function = self.create_done_function(
            done_function_type=done_function_type
        )

    @classmethod
    def create_reward_function(cls, reward_function_type: str) -> RewardFunction:
        if reward_function_type == "default":
            return DefaultRewardFunction()
        else:
            raise ValueError(
                f"Unexpected value for reward_function_type, got {reward_function_type}. "
                f"Possible values: 'default'"
            )

    @classmethod
    def create_done_function(cls, done_function_type: str) -> DoneFunction:
        if done_function_type == "default":
            return DefaultDoneFunction()
        else:
            raise ValueError(
                f"Unexpected value for done_function_type, "
                f"got {done_function_type}. Possible values: 'default'"
            )

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment.

        Args:
            key: needed for placing mines.

        Returns:
            state: State corresponding to the new state of the environment,
            timestep: TimeStep corresponding to the first timestep returned by the
                environment.
        """
        board = jnp.full(
            shape=(self.board_height, self.board_width),
            fill_value=UNEXPLORED_ID,
            dtype=jnp.int32,
        )
        step_count = jnp.int32(0)
        flat_mine_locations = create_flat_mine_locations(
            key=key,
            board_height=self.board_height,
            board_width=self.board_width,
            num_mines=self.num_mines,
        )
        state = State(
            board=board,
            step_count=step_count,
            key=key,
            flat_mine_locations=flat_mine_locations,
        )
        observation = state_to_observation(state=state, num_mines=self.num_mines)
        timestep = restart(observation=observation)
        return state, timestep

    def step(self, state: State, action: Action) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the height and width of the square to be explored.

        Returns:
            next_state: State corresponding to the next state of the environment,
            next_timestep: TimeStep corresponding to the timestep returned by the environment.
        """
        board = state.board
        action_height, action_width = action
        board = board.at[action_height, action_width].set(
            count_adjacent_mines(state=state, action=action)
        )
        step_count = state.step_count + 1
        next_state = State(
            board=board,
            step_count=step_count,
            key=state.key,
            flat_mine_locations=state.flat_mine_locations,
        )
        reward = self.reward_function(state, action)
        done = self.done_function(state, next_state, action)
        next_observation = state_to_observation(
            state=next_state, num_mines=self.num_mines
        )
        next_timestep = lax.cond(
            done,
            termination,
            transition,
            reward,
            next_observation,
        )
        return next_state, next_timestep

    def observation_spec(self) -> ObservationSpec:
        """Returns the observation spec containing the board, number of mines, and step count.

        Returns:
            observation_spec: ObservationSpec tree of the board, number of mines,
                and step count spec.
        """
        return ObservationSpec(
            board=specs.BoundedArray(
                shape=(self.board_height, self.board_width),
                dtype=jnp.int32,
                minimum=-1,
                maximum=8,
                name="board",
            ),
            action_mask=specs.BoundedArray(
                shape=(self.board_height, self.board_width),
                dtype=jnp.bool_,
                minimum=False,
                maximum=True,
                name="action_mask",
            ),
            num_mines=specs.BoundedArray(
                shape=(),
                dtype=jnp.int32,
                minimum=0,
                maximum=self.board_height * self.board_width - 1,
                name="num_mines",
            ),
            step_count=specs.BoundedArray(
                shape=(),
                dtype=jnp.int32,
                minimum=0,
                maximum=self.board_height * self.board_width - self.num_mines,
                name="step_count",
            ),
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec.
        An action consists of the height and width of the square to be explored.

        Returns:
            action_spec: specs.MultiDiscreteArray object
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.array([self.board_height, self.board_width]),
            name="action",
            dtype=jnp.int32,
        )

    def render(self, state: State) -> str:
        """Renders a given state.

        Args:
            state: State object corresponding to the new state of the environment.

        Returns:
            human-readable string displaying the current state of the game.

        """
        message = f"Board: {state.board}\n"
        message += f"Number of mines: {str(state.flat_mine_locations.shape[0])}"
        message += f"Step count: {str(state.step_count)}"
        return message
