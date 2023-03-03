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

from typing import Any, List, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.animation
import matplotlib.artist
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

import jumanji
import jumanji.environments
from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.snake.types import Observation, Position, State
from jumanji.types import TimeStep, restart, termination, transition


class Snake(Environment[State]):
    """A JAX implementation of the 'Snake' game.

    - observation: `Observation`
        - grid: jax array (float) of shape (num_rows, num_cols, 5)
            feature maps that include information about the fruit, the snake head, its body and
            tail.
            - body: 2D map with 1. where a body cell is present, else 0.
            - head: 2D map with 1. where the snake's head is located, else 0.
            - tail: 2D map with 1. where the snake's tail is located, else 0.
            - fruit: 2D map with 1. where the fruit is located, else 0.
            - norm_body_state: 2D map with a float between 0. and 1. for each body cell in the
                decreasing order from head to tail.
        - step_count: jax array (int32) of shape ()
            current number of steps in the episode.
        - action_mask: jax array (bool) of shape (4,)
            array specifying which directions the snake can move in from its current position.

    - action: jax array (int32) of shape()
        [0,1,2,3] -> [Up, Right, Down, Left].

    - reward: jax array (float) of shape ()
        1.0 if a fruit is eaten, otherwise 0.0.

    - episode termination:
        - if no action can be performed, i.e. the snake is surrounded.
        - if the time limit is reached.
        - if an invalid action is taken, the snake exits the grid or bumps into itself.

    - state: `State`
        - key: jax array (uint32) of shape (2,)
            random key used to sample a new fruit when one is eaten.
        - body: jax array (bool) of shape (num_rows, num_cols)
            array indicating the snake's body cells.
        - body_state: jax array (int32) of shape (num_rows, num_cols)
            array ordering the snake's body cells, in decreasing order from head to tail.
        - head_position: `Position` (int32) of shape ()
            position of the snake's head on the 2D grid.
        - tail: jax array (bool) of shape (num_rows, num_cols)
            array indicating the snake's tail.
        - fruit_position: `Position` (int32) of shape ()
            position of the fruit on the 2D grid.
        - length: jax array (int32) of shape ()
            current length of the snake.
        - step_count: jax array (int32) of shape ()
            current number of steps in the episode.
        - action_mask: jax array (bool) of shape (4,)
            array specifying which directions the snake can move in from its current position.
    """

    FIGURE_NAME = "Snake"
    FIGURE_SIZE = (6.0, 6.0)
    MOVES = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], jnp.int32)

    def __init__(self, num_rows: int = 12, num_cols: int = 12, time_limit: int = 4000):
        """Instantiates a `Snake` environment.

        Args:
            num_rows: number of rows of the 2D grid.
                Defaults to 12.
            num_cols: number of columns of the 2D grid.
                Defaults to 12.
            time_limit: horizon of an episode, i.e. number of environment steps before
                the episode ends. Defaults to 4000.
        """
        super().__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.board_shape = (num_rows, num_cols)
        self.time_limit = time_limit

        # You must store the created Animation in a variable that lives as long as the animation
        # should run. Otherwise, the animation will get garbage-collected.
        self._animation: Optional[matplotlib.animation.Animation] = None

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Snake environment:",
                f" - num_rows: {self.num_rows}",
                f" - num_cols: {self.num_cols}",
                f" - time_limit: {self.time_limit}",
            ]
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment.

        Args:
            key: random key used to sample the snake and fruit positions.

        Returns:
             state: `State` object corresponding to the new state of the environment.
             timestep: `TimeStep` object corresponding to the first timestep returned by the
                environment.
        """
        key, snake_key, fruit_key = jax.random.split(key, 3)
        # Sample Snake's head position.
        head_coordinates = jax.random.randint(
            snake_key,
            shape=(2,),
            minval=jnp.zeros(2, int),
            maxval=jnp.array(self.board_shape),
        )
        head_position = Position(*tuple(head_coordinates))

        body = jnp.zeros(self.board_shape, bool).at[tuple(head_position)].set(True)
        tail = body
        body_state = body.astype(jnp.int32)
        fruit_position = self._sample_fruit_coord(body, fruit_key)
        state = State(
            key=key,
            body=body,
            body_state=body_state,
            head_position=head_position,
            tail=tail,
            fruit_position=fruit_position,
            length=jnp.array(1, jnp.int32),
            step_count=jnp.array(0, jnp.int32),
            action_mask=self._get_action_mask(head_position, body_state),
        )
        timestep = restart(observation=self._state_to_observation(state))
        return state, timestep

    def step(
        self, state: State, action: chex.Numeric
    ) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: `State` object containing the dynamics of the environment.
            action: Array containing the action to take:
                - 0: move up.
                - 1: move to the right.
                - 2: move down.
                - 3: move to the left.

        Returns:
            state, timestep: next state of the environment and timestep to be observed.
        """
        is_valid = state.action_mask[action]
        key, fruit_key = jax.random.split(state.key)

        head_position = self._update_head_position(state.head_position, action)

        fruit_eaten = head_position == state.fruit_position

        length = state.length + fruit_eaten

        body_state_without_head = jax.lax.select(
            fruit_eaten,
            state.body_state,
            jnp.clip(state.body_state - 1, 0),
        )
        body_state = body_state_without_head.at[tuple(head_position)].set(length)

        body = body_state > 0

        tail = body_state == 1

        fruit_position = jax.lax.cond(
            fruit_eaten,
            self._sample_fruit_coord,
            lambda *_: state.fruit_position,
            body,
            fruit_key,
        )
        step_count = state.step_count + 1
        next_state = State(
            key=key,
            body=body,
            body_state=body_state,
            head_position=head_position,
            tail=tail,
            fruit_position=fruit_position,
            length=length,
            step_count=state.step_count + 1,
            action_mask=self._get_action_mask(head_position, body_state),
        )

        snake_completed = jnp.all(body)
        done = ~is_valid | snake_completed | (step_count >= self.time_limit)

        reward = jnp.asarray(fruit_eaten, float)
        observation = self._state_to_observation(next_state)

        timestep = jax.lax.cond(
            done,
            termination,
            transition,
            reward,
            observation,
        )
        return next_state, timestep

    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec.

        Returns:
            Spec for the `Observation` whose fields are:
            - grid: BoundedArray (float) of shape (num_rows, num_cols, 5).
            - step_count: DiscreteArray (num_values = time_limit) of shape ().
            - action_mask: BoundedArray (bool) of shape (4,).
        """
        grid = specs.BoundedArray(
            shape=(self.num_rows, self.num_cols, 5),
            minimum=0.0,
            maximum=1.0,
            dtype=float,
            name="grid",
        )
        step_count = specs.DiscreteArray(
            self.time_limit, dtype=jnp.int32, name="step_count"
        )
        action_mask = specs.BoundedArray(
            shape=(4,),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            grid=grid,
            step_count=step_count,
            action_mask=action_mask,
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec. 4 actions: [0,1,2,3] -> [Up, Right, Down, Left].

        Returns:
            action_spec: a `specs.DiscreteArray` spec.
        """
        return specs.DiscreteArray(4, name="action")

    def _state_to_observation(self, state: State) -> Observation:
        """Maps an environment state to an observation.

        Args:
            state: `State` object containing the dynamics of the environment.

        Returns:
            The observation derived from the state.
        """
        body = state.body
        head = jnp.zeros_like(body).at[tuple(state.head_position)].set(True)
        tail = state.tail
        fruit = jnp.zeros_like(body).at[tuple(state.fruit_position)].set(True)
        norm_body_state = state.body_state / jnp.maximum(1, state.body_state.max())
        grid = jnp.concatenate(
            jax.tree_util.tree_map(
                lambda x: x[..., None], [body, head, tail, fruit, norm_body_state]
            ),
            axis=-1,
            dtype=float,
        )

        return Observation(
            grid=grid,
            step_count=state.step_count,
            action_mask=state.action_mask,
        )

    def _sample_fruit_coord(
        self,
        body: chex.Array,
        key: chex.PRNGKey,
    ) -> Position:
        """Sample the coordinates of a new fruit at a location where the snake is not.

        Args:
            body: array indicating the snake's body cells.
            key: random key to generate a random fruit position.

        Returns:
            Position(row, col) corresponding to the new fruit coordinates.
        """
        fruit_index = jax.random.choice(
            key,
            jnp.arange(self.num_rows * self.num_cols),
            p=~body.flatten(),
        )
        row, col = jnp.divmod(fruit_index, self.num_cols)
        return Position(row=row, col=col)

    def _get_action_mask(
        self,
        head_position: Position,
        body_state: chex.Array,
    ) -> chex.Array:
        """Checks whether the episode is over or not.

        Args:
            head_position: Position of the snake's head.
            body_state: array ordering the snake's body cells.

        Returns:
            action_mask: array (bool) of shape (4,).
        """

        def is_valid(move: chex.Array) -> chex.Array:
            new_head_position = head_position + Position(*tuple(move))
            outside_board = (
                (new_head_position.row < 0)
                | (new_head_position.row >= self.num_rows)
                | (new_head_position.col < 0)
                | (new_head_position.col >= self.num_cols)
            )
            body_state_without_head = jnp.clip(body_state - 1, 0)
            head_bumps_body = body_state_without_head[tuple(new_head_position)] > 0
            return ~outside_board & ~head_bumps_body

        action_mask = jax.vmap(is_valid)(self.MOVES)
        return action_mask

    def _update_head_position(
        self, head_position: Position, action: chex.Numeric
    ) -> Position:
        """Give the new head position after taking an action.

        Args:
            head_position: `Position` of the snake's head.
            action: integer that tells in which direction to go.

        Returns:
            New head position after taking the action.
        """
        # Possible moves are: Up, Right, Down, Left.
        row_move, col_move = self.MOVES[action]
        move_position = Position(row=row_move, col=col_move)
        next_head_position = Position(*tuple(head_position)) + move_position
        return next_head_position

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
        ax.plot([0, 0], [0, self.num_rows], "-k", lw=2)
        ax.plot([0, self.num_cols], [self.num_rows, self.num_rows], "-k", lw=2)
        ax.plot([self.num_cols, self.num_cols], [self.num_rows, 0], "-k", lw=2)
        ax.plot([self.num_cols, 0], [0, 0], "-k", lw=2)

    def _create_entities(self, state: State) -> List[matplotlib.patches.Patch]:
        """Loop over the different cells and draws corresponding shapes in the ax object."""
        patches = []
        linewidth = (
            min(
                n * size
                for n, size in zip((self.num_rows, self.num_cols), self.FIGURE_SIZE)
            )
            / 44.0
        )
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "", ["yellowgreen", "forestgreen"]
        )
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if state.body_state[row, col]:
                    body_cell_patch = Rectangle(
                        (col, self.num_rows - 1 - row),
                        1,
                        1,
                        edgecolor=cmap(1),
                        facecolor=cmap(state.body_state[row, col] / state.length),
                        fill=True,
                        lw=linewidth,
                    )
                    patches.append(body_cell_patch)
        head_patch = Circle(
            (
                state.head_position[1] + 0.5,
                self.num_rows - 1 - state.head_position[0] + 0.5,
            ),
            0.3,
            edgecolor=cmap(0.5),
            facecolor=cmap(0),
            fill=True,
            lw=linewidth,
        )
        patches.append(head_patch)
        fruit_patch = Circle(
            (
                state.fruit_position[1] + 0.5,
                self.num_rows - 1 - state.fruit_position[0] + 0.5,
            ),
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
