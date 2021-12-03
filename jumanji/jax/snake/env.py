from typing import Tuple, Union

import jax.numpy as jnp
from chex import Array, PRNGKey
from dm_env import specs
from jax import lax, random

from jumanji.jax.env import JaxEnv
from jumanji.jax.snake import utils as snake_utils
from jumanji.jax.snake.types import Action, Position, State
from jumanji.jax.types import TimeStep, restart, termination, transition


class Snake(JaxEnv[State]):
    """
    A JAX implementation of the 'Snake' game.

    - observation: jax array (float) of shape (5, n_rows, n_cols):
        - body: binary encoding (1. where a body cell is present, else 0.)
        - head: one-hot encoding (1. where the snake's head is, else 0.)
        - tail: one-hot encoding (1. where the snake's tail is, else 0.)
        - fruit: one-hot encoding (1. where the fruit is, else 0.)
        - body_state: float encoding (between 0. and 1.)

    - reward: 1 if fruit is eaten, 0 otherwise.

    - episode termination: if snake bumps into its body, or goes outside the board,
        or the horizon is reached, or the grid is fully filled by the snake, which means
        it has eaten all the fruits.

    - state: State:
        - key: jax array (uint) of shape (2,) used for the random generation of fruits.
        - body_state: jax array (float) of shape (n_rows, n_cols) containing integers from
            0 to length corresponding to the order of the snake body cells. 0 means no body,
            1 is the tail, and {length} is the head.
        - head_pos: Position(row, col), position of the snake's head on the grid.
        - fruit_pos: Position(row, col), position of the fruit on the grid.
        - length: int, number of body cells.
        - step: int, step number in the episode.
    """

    def __init__(self, n_rows: int = 12, n_cols: int = 12, time_limit: int = 5000):
        """Instantiates a Snake environment.

        Args:
            n_rows: number of rows of the 2D grid.
            n_cols: number of columns of the 2D grid.
            time_limit: horizon of an episode, i.e. number of environment steps before
                the episode ends.

        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.board_shape = (n_rows, n_cols)
        self.time_limit = time_limit

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Snake environment:",
                f" - n_rows: {self.n_rows}",
                f" - n_cols: {self.n_cols}",
                f" - time_limit: {self.time_limit}",
            ]
        )

    def observation_spec(self) -> specs.Array:
        """Returns the observation spec.

        Returns:
            observation_spec: dm_env.specs object
        """
        return specs.Array(shape=(5, *self.board_shape), dtype=jnp.float32)

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec. 4 actions: [0,1,2,3] -> [Up, Right, Down, Left].

        Returns:
            action_spec: dm_env.specs object
        """
        return specs.DiscreteArray(4, name="action")

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep]:
        """Resets the environment.

        Args:
            key: random key used to reset the environment since it is stochastic.

        Returns:
            state, timestep: Tuple[State, TimeStep] containing the new state of the environment,
                as well as the first timestep.
        """
        key, body_key, fruit_key = random.split(key, 3)
        body = jnp.zeros(self.board_shape, bool)
        start_coord = random.randint(
            body_key,
            shape=(2,),
            minval=jnp.zeros(2, int),
            maxval=jnp.array(self.board_shape),
        )
        body = body.at[tuple(start_coord)].set(True)
        tail = body
        head = body
        fruit_coord = Snake._sample_fruit_coord(body, fruit_key, self.board_shape)
        fruit_pos = snake_utils.position_from_coordinates(fruit_coord)
        start_pos = snake_utils.position_from_coordinates(start_coord)
        fruit = jnp.zeros(self.board_shape, bool).at[tuple(fruit_coord)].set(True)
        body_state = body.astype(float)
        obs = jnp.stack([body, head, tail, fruit, body_state]).astype(float)
        state = State(
            key=key,
            body_state=body_state,
            head_pos=start_pos,
            fruit_pos=fruit_pos,
            length=jnp.int32(1),
            step=jnp.int32(0),
        )
        timestep = restart(observation=obs)
        return state, timestep

    def step(self, state: State, action: Action) -> Tuple[State, TimeStep]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the action number to take.
                - 0 move up
                - 1 move to the right
                - 2 move down
                - 3 move to the left

        Returns:
            State and TimeStep observed after executing the action.

        """
        state, next_body_no_head, fruit_eaten = self._update_state(state, action)
        timestep = self._state_to_timestep(state, next_body_no_head, fruit_eaten)
        return state, timestep

    def _update_state(self, state: State, action: Action) -> Tuple[State, Array, Array]:
        """Update the environment state by taking an action.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the action number to take.

        Returns:
            state: environment state after taking one step.
            next_body_no_head: jnp array (bool), binary encoding after a step in the environment
                (True where a body cell of the snake without its head is present, else False).
            fruit_eaten: boolean True if a fruit has been eaten while taking the step, else False.

        """
        next_key, fruit_key = random.split(state.key)

        # Update head
        next_head_pos = self.update_head_pos(state.head_pos, action)
        fruit_eaten = next_head_pos == state.fruit_pos

        # Update length
        next_length = lax.cond(
            fruit_eaten,
            lambda x: x + 1,
            lambda x: x,
            state.length,
        )

        # Update body
        next_body_state_no_head = lax.cond(
            fruit_eaten,
            lambda x: x,
            lambda x: jnp.clip(x - 1, 0),
            state.body_state,
        )
        next_body_no_head = next_body_state_no_head > 0
        next_body = next_body_no_head.at[tuple(next_head_pos)].set(True)
        next_body_state = next_body_state_no_head.at[tuple(next_head_pos)].set(
            next_length
        )

        # Update fruit
        next_fruit_coord = lax.select(
            fruit_eaten,
            Snake._sample_fruit_coord(next_body, fruit_key, self.board_shape),
            jnp.array(tuple(state.fruit_pos)),
        )
        next_fruit_pos = snake_utils.position_from_coordinates(next_fruit_coord)

        # Update step
        next_step = state.step + 1

        next_state = State(
            key=next_key,
            body_state=next_body_state,
            head_pos=next_head_pos,
            fruit_pos=next_fruit_pos,
            length=next_length,
            step=next_step,
        )

        return next_state, next_body_no_head, fruit_eaten

    def _state_to_timestep(
        self, state: State, next_body_no_head: Array, fruit_eaten: bool
    ) -> TimeStep:
        """Maps an environment state to an observation. To do this, one needs the body without the
        head, as well as whether a fruit has been eaten.

        Args:
            state: State of the environment after taking a step.
            next_body_no_head: jnp array (bool), binary encoding after a step in the environment
                (True where a body cell of the snake without its head is present, else False).
            fruit_eaten: boolean True if a fruit has been eaten while taking the step, else False.

        Returns:
            timestep containing the new observation.

        """
        next_head_pos = state.head_pos
        next_step = state.step
        next_body_state = state.body_state
        next_body = next_body_state > 0
        reward = lax.select(fruit_eaten, jnp.float32(1), jnp.float32(0))
        done = self._is_done(
            next_head_pos, next_body_no_head, next_body_state, next_step
        )
        next_tail = next_body_state == 1
        next_fruit = (
            jnp.zeros(self.board_shape, bool).at[tuple(state.fruit_pos)].set(True)
        )
        next_head = (
            jnp.zeros(self.board_shape, bool).at[tuple(state.head_pos)].set(True)
        )
        next_obs = jnp.array(
            [
                next_body,
                next_head,
                next_tail,
                next_fruit,
                next_body_state / jnp.maximum(1, next_body_state.max()),
            ],
            dtype=jnp.float32,
        )

        timestep: TimeStep = lax.cond(
            done,
            lambda _: termination(reward=reward, observation=next_obs),
            lambda _: transition(reward=reward, observation=next_obs),
            None,
        )
        return timestep

    @staticmethod
    def _sample_fruit_coord(
        body: Array, key: PRNGKey, board_shape: Tuple[int, int]
    ) -> Array:
        """Internal function that samples the coordinates of a new fruit.

        Args:
            body: array containing the body of the snake, accessed from state.body_state.
            key: random key to generate a random fruit position.
            board_shape: tuple(n_rows, n_cols) used to sample a valid position.

        Returns:
            array of tuple (row, col) corresponding to the new fruit coordinates.
        """
        n_rows, n_cols = board_shape
        fruit_index = random.choice(
            key, jnp.arange(n_rows * n_cols), p=jnp.logical_not(body).flatten()
        )
        row = fruit_index // n_cols
        col = fruit_index % n_cols
        return jnp.array((row, col))

    def _is_done(
        self,
        next_head_pos: Position,
        next_body_no_head: Array,
        next_body: Array,
        next_step: Array,
    ) -> Union[bool, Array]:
        """Checks whether the episode is over or not.

        Args:
            next_head_pos: Position of the head after taking the action.
            next_body_no_head: array of cells other than the head after moving.
            next_body: array of body cells including the new head position.
            next_step: number of step in the episode to check if the horizon is attained.

        Returns:
            boolean True if episode is over, else False.
        """
        done = jnp.any(
            jnp.array(
                [
                    self._head_outside_board(next_head_pos),
                    self._head_bumps_body(next_body_no_head, next_head_pos),
                    self._snake_completed(next_body),
                    next_step >= self.time_limit,
                ]
            )
        )
        return done

    def _head_outside_board(self, head_pos: Position) -> Union[bool, Array]:
        """Checks if the head has gone beyond the borders of the grid.

        Args:
            head_pos: Position of the head.

        Returns:
            True if the head is outside the grid, else False.
        """
        row, col = head_pos.row, head_pos.col
        return jnp.any(
            jnp.array([row < 0, row >= self.n_rows, col < 0, col >= self.n_cols])
        )

    @staticmethod
    def _head_bumps_body(
        next_body_no_head: Array, next_head_pos: Position
    ) -> Union[bool, Array]:
        """Checks if the head bumps into the body.

        Args:
            next_body_no_head: array of cells other than the head after moving.
            next_head_pos: Position of the head after taking the action.

        Returns:
            True if the head bumps into the body, else False.
        """
        return next_body_no_head[tuple(next_head_pos)]

    @staticmethod
    def _snake_completed(next_body: Array) -> Union[bool, Array]:
        """Checks if the Snake board is full, hence meaning the game is completed.

        Args:
            next_body: array of body cells including the new head position.

        Returns:
            True if the board is complete.
        """
        return jnp.all(next_body)

    @staticmethod
    def update_head_pos(head_pos: Position, action: Action) -> Position:
        """Starting from a head position, gives the new one after taking an action.

        Args:
            head_pos: Position of the current head.
            action: integer that tells which direction to go.

        Returns:
            New Position after taking the action.
        """
        row, col = tuple(head_pos)

        # corresponding to up right down left
        position_change_branches = [
            lambda row_col: jnp.array((row_col[0] - 1, row_col[1])),
            lambda row_col: jnp.array((row_col[0], row_col[1] + 1)),
            lambda row_col: jnp.array((row_col[0] + 1, row_col[1])),
            lambda row_col: jnp.array((row_col[0], row_col[1] - 1)),
        ]
        next_head_coord = lax.switch(action, position_change_branches, (row, col))
        next_head_pos = snake_utils.position_from_coordinates(next_head_coord)
        return next_head_pos
