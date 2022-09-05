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

from typing import Any, List, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import matplotlib
import matplotlib.animation
import matplotlib.artist
import matplotlib.pyplot as plt
from chex import Array, PRNGKey
from jax import lax, random
from matplotlib.patches import Circle, Rectangle

import jumanji.environments
from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.games.snake import utils as snake_utils
from jumanji.environments.games.snake.types import Position, State
from jumanji.types import Action, TimeStep, restart, termination, transition


class Snake(Environment[State]):
    """A JAX implementation of the 'Snake' game.

    - observation: jax array (float) of shape (n_rows, n_cols, 5):
        - body: binary encoding (1. where a body cell is present, else 0.)
        - head: one-hot encoding (1. where the snake's head is, else 0.)
        - tail: one-hot encoding (1. where the snake's tail is, else 0.)
        - fruit: one-hot encoding (1. where the fruit is, else 0.)
        - body_state: float encoding (between 0. and 1.)

    - action: jax array (int) of shape(): [0,1,2,3] -> [Up, Right, Down, Left].

    - reward: 1 if fruit is eaten, 0 otherwise.

    - episode termination (if any):
        - the snake bumps into its body.
        - the snake goes outside the board.
        - the horizon is reached.
        - the grid is fully filled by the snake, which means it has eaten all the fruits.

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

    FIGURE_NAME = "Snake"
    FIGURE_SIZE = (6.0, 6.0)

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

        # You must store the created Animation in a variable that lives as long as the animation
        # should run. Otherwise the animation will get garbage-collected.
        self._animation: Optional[matplotlib.animation.Animation] = None

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
        return specs.Array(
            shape=(*self.board_shape, 5), dtype=jnp.float32, name="observation"
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec. 4 actions: [0,1,2,3] -> [Up, Right, Down, Left].

        Returns:
            action_spec: dm_env.specs object
        """
        return specs.DiscreteArray(4, name="action")

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep[Array]]:
        """Resets the environment.

        Args:
            key: random key used to reset the environment since it is stochastic.

        Returns:
            state: State object corresponding to the new state of the environment,
            timestep: TimeStep object corresponding the first timestep returned by the environment,
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
        obs = jnp.stack([body, head, tail, fruit, body_state], axis=-1).astype(float)
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
                - 0: move up.
                - 1: move to the right.
                - 2: move down.
                - 3: move to the left.

        Returns:
            state: State object corresponding to the next state of the environment.
            timestep: TimeStep object corresponding the timestep returned by the environment.
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

        # build the observation array
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

        # convert to the HWC format
        next_obs = jnp.transpose(next_obs, axes=(1, 2, 0))

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

    def render(self, state: State) -> None:
        """Render frames of the environment for a given state using matplotlib.

        Args:
            state: State object containing the current dynamics of the environment.

        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        self._draw(ax, state)
        self._update_display(fig)

    def close(self) -> None:
        """Perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        plt.close(self.FIGURE_NAME)

    def animation(
        self,
        states: Sequence[State],
        interval: int = 200,
        blit: bool = False,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of states.

        Args:
            states: sequence of `State` corresponding to subsequent timesteps.
            interval: delay between frames in milliseconds, default to 200.
            blit: whether to use blitting to optimize drawing, default to False.
                Note: when using blitting, any animated artists will be drawn according to their
                zorder. However, they will be drawn on top of any previous artists, regardless
                of their zorder.

        Returns:
            animation that can export to gif, mp4, or render with HTML.

        Example:
            ```python
            env = Snake(6, 6)
            states = []
            state, _ = env.reset(jax.random.PRNGKey(0))
            states.append(state)
            state, _ = env.step(state, env.action_spec().generate_value())
            states.append(state)
            state, _ = env.step(state, env.action_spec().generate_value())
            states.append(state)
            state, _ = env.step(state, env.action_spec().generate_value())
            states.append(state)
            animation = env.animation(states)
            animation.save("anim.gif", writer=matplotlib.animation.FFMpegWriter(fps=10), dpi=60)
            ```
        """
        fig, ax = plt.subplots(num=f"{self.FIGURE_NAME}Anim", figsize=self.FIGURE_SIZE)
        self._draw_board(ax)
        plt.close(fig)

        patches: List[matplotlib.patches.Patch] = []

        def animate(state: State) -> Any:
            while patches:
                patches.pop().remove()
            patches.extend(self._create_entities(state))
            for patch in patches:
                ax.add_patch(patch)

        matplotlib.rc("animation", html="jshtml")

        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            animate,
            frames=states,
            blit=blit,
            interval=interval,
        )
        return self._animation

    def _get_fig_ax(self) -> Tuple[plt.Figure, plt.Axes]:
        exists = plt.fignum_exists(self.FIGURE_NAME)
        if exists:
            fig = plt.figure(self.FIGURE_NAME)
            ax = fig.get_axes()[0]
        else:
            fig = plt.figure(self.FIGURE_NAME, figsize=self.FIGURE_SIZE)
            fig.set_tight_layout({"pad": False, "w_pad": 0.0, "h_pad": 0.0})
            if not plt.isinteractive():
                fig.show()
            ax = fig.add_subplot()
        return fig, ax

    def _draw(self, ax: plt.Axes, state: State) -> None:
        ax.clear()
        self._draw_board(ax)
        for patch in self._create_entities(state):
            ax.add_patch(patch)

    def _draw_board(self, ax: plt.Axes) -> None:
        # Draw the square box that delimits the board.
        ax.axis("off")
        ax.plot([0, 0], [0, self.n_rows], "-k", lw=2)
        ax.plot([0, self.n_cols], [self.n_rows, self.n_rows], "-k", lw=2)
        ax.plot([self.n_cols, self.n_cols], [self.n_rows, 0], "-k", lw=2)
        ax.plot([self.n_cols, 0], [0, 0], "-k", lw=2)

    def _create_entities(self, state: State) -> List[matplotlib.patches.Patch]:
        """Loop over the different cells and draws corresponding shapes in the ax object."""
        patches = []
        linewidth = (
            min(
                n * size
                for n, size in zip((self.n_rows, self.n_cols), self.FIGURE_SIZE)
            )
            / 44.0
        )
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "", ["yellowgreen", "forestgreen"]
        )
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                if state.body_state[row, col]:
                    body_cell_patch = Rectangle(
                        (col, self.n_rows - 1 - row),
                        1,
                        1,
                        edgecolor=cmap(1),
                        facecolor=cmap(state.body_state[row, col] / state.length),
                        fill=True,
                        lw=linewidth,
                    )
                    patches.append(body_cell_patch)
        head_patch = Circle(
            (state.head_pos[1] + 0.5, self.n_rows - 1 - state.head_pos[0] + 0.5),
            0.3,
            edgecolor=cmap(0.5),
            facecolor=cmap(0),
            fill=True,
            lw=linewidth,
        )
        patches.append(head_patch)
        fruit_patch = Circle(
            (state.fruit_pos[1] + 0.5, self.n_rows - 1 - state.fruit_pos[0] + 0.5),
            0.2,
            edgecolor="brown",
            facecolor="lightcoral",
            fill=True,
            lw=linewidth,
        )
        patches.append(fruit_patch)
        return patches

    def _update_display(self, fig: plt.Figure) -> None:
        if plt.isinteractive():
            # Required to update render when using Jupyter Notebook.
            fig.canvas.draw()
            if jumanji.environments.is_colab():
                plt.show(self.FIGURE_NAME)
        else:
            # Required to update render when not using Jupyter Notebook.
            fig.canvas.draw_idle()
            # Block for 0.5 seconds.
            fig.canvas.start_event_loop(0.5)

    def _clear_display(self) -> None:
        if jumanji.environments.is_colab():
            import IPython.display

            IPython.display.clear_output(True)
