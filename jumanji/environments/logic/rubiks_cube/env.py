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

from typing import List, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt

import jumanji.environments
from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.logic.rubiks_cube.constants import (
    DEFAULT_STICKER_COLORS,
    CubeMovementAmount,
    Face,
)
from jumanji.environments.logic.rubiks_cube.reward import RewardFn, SparseRewardFn
from jumanji.environments.logic.rubiks_cube.types import Cube, Observation, State
from jumanji.environments.logic.rubiks_cube.utils import (
    generate_all_moves,
    is_solved,
    make_solved_cube,
)
from jumanji.types import TimeStep, restart, termination, transition


class RubiksCube(Environment[State]):
    """A JAX implementation of the Rubik's Cube with a configurable cube size (by default, 3) and
    number of scrambles at reset.

    - observation: `Observation`
        - cube: jax array (int8) of shape (6, cube_size, cube_size):
            each cell contains the index of the corresponding colour of the sticker in the scramble.
        - step_count: jax array (int32) of shape ():
            specifies how many timesteps have elapsed since environment reset.

    - action:
        multi discrete array containing the move to perform (face, depth, and direction).

    - reward: jax array (float) of shape ():
        by default, 1.0 if cube is solved, otherwise 0.0.

    - episode termination:
        if either the cube is solved or a time limit is reached.

    - state: `State`
        - cube: jax array (int8) of shape (6, cube_size, cube_size):
            each cell contains the index of the corresponding colour of the sticker in the scramble.
        - step_count: jax array (int32) of shape ():
            specifies how many timesteps have elapsed since environment reset.
        - key: jax array (uint) of shape (2,) used for seeding the sampling for scrambling on
            reset.
        - action_history: jax array (int32) of shape (num_scrambles_on_reset + time_limit, 3):
            indicates the entire history of applied moves (including those taken on scrambling the
            cube in the environment reset). This is useful for debugging purposes, providing a
            method to solve the cube from any position without relying on the agent, by just
            inverting the action history. The first axis indexes over the length of the sequence
            The second axis indexes over the component of the action (face, depth, amount). The
            number of scrambles applied for each state is given by
            `env.num_scrambles_on_reset + state.step_count`.

    ```python
    from jumanji.environments import RubiksCube
    env = RubiksCube()
    key = jax.random.key(0)
    state, timestep = jax.jit(env.reset)(key)
    env.render(state)
    action = env.action_spec().generate_value()
    state, timestep = jax.jit(env.step)(state, action)
    env.render(state)
    ```
    """

    def __init__(
        self,
        cube_size: int = 3,
        time_limit: int = 200,
        num_scrambles_on_reset: int = 100,
        reward_fn: Optional[RewardFn] = None,
        sticker_colors: Optional[list] = None,
    ):
        """Instantiate a `RubiksCube` environment.

        Args:
            cube_size: the size of the cube, i.e. length of an edge. Defaults to 3.
            time_limit: the number of steps allowed before an episode terminates. Defaults to 200.
            num_scrambles_on_reset: the number of scrambles done from a solved Rubik's Cube in the
                generation of a random instance. The lower, the closer to a solved cube the reset
                state is. Defaults to 100.
            reward_fn: `RewardFn` whose `__call__` method computes the reward given the new state.
                Implemented options are [`SparseRewardFn`]. Defaults to `SparseRewardFn`.
            sticker_colors: colors used in rendering the faces of the rubiks cube.
                Defaults to `DEFAULT_STICKER_COLORS`.
        """
        if cube_size < 2:
            raise ValueError(
                f"Cannot meaningfully construct a cube smaller than 2x2x2, "
                f"but received cube_size={cube_size}"
            )
        if time_limit <= 0:
            raise ValueError(
                f"The time_limit must be positive, but received time_limit={time_limit}"
            )
        if num_scrambles_on_reset < 0:
            raise ValueError(
                f"The num_scrambles_on_reset must be non-negative, "
                f"but received num_scrambles_on_reset={num_scrambles_on_reset}"
            )
        self.cube_size = cube_size
        self.time_limit = time_limit
        self.num_scrambles_on_reset = num_scrambles_on_reset
        self.reward_function = reward_fn or SparseRewardFn()
        sticker_colors = sticker_colors or DEFAULT_STICKER_COLORS
        self.sticker_colors_cmap = matplotlib.colors.ListedColormap(sticker_colors)
        self.num_actions = len(Face) * (cube_size // 2) * len(CubeMovementAmount)
        self.all_moves = generate_all_moves(cube_size=cube_size)

        self.figure_name = f"{cube_size}x{cube_size}x{cube_size} Rubik's Cube"
        self.figure_size = (6.0, 6.0)

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment.

        Args:
            key: needed for scramble.

        Returns:
            state: `State` corresponding to the new state of the environment.
            timestep: `TimeStep` corresponding to the first timestep returned by the
                environment.
        """
        key, scramble_key = jax.random.split(key)
        flat_actions_in_scramble = jax.random.randint(
            scramble_key,
            minval=0,
            maxval=self.num_actions,
            shape=(self.num_scrambles_on_reset,),
            dtype=jnp.int32,
        )
        cube = self._scramble_solved_cube(
            flat_actions_in_scramble=flat_actions_in_scramble
        )
        action_history = jnp.zeros(
            shape=(self.num_scrambles_on_reset + self.time_limit, 3), dtype=jnp.int32
        )
        action_history = action_history.at[: self.num_scrambles_on_reset].set(
            self._unflatten_action(flat_actions_in_scramble).transpose()
        )
        step_count = jnp.array(0, jnp.int32)
        state = State(
            cube=cube,
            step_count=step_count,
            key=key,
            action_history=action_history,
        )
        observation = self._state_to_observation(state=state)
        timestep = restart(observation=observation)
        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: `State` object containing the dynamics of the environment.
            action: `Array` of shape (3,) indicating the face to move, depth of the move, and the
                amount to move by.

        Returns:
            next_state: `State` corresponding to the next state of the environment.
            next_timestep: `TimeStep` corresponding to the timestep returned by the environment.
        """
        flat_action = self._flatten_action(action)
        cube = self._rotate_cube(cube=state.cube, flat_action=flat_action)
        action_history = state.action_history.at[
            self.num_scrambles_on_reset + state.step_count
        ].set(action)
        step_count = state.step_count + 1
        next_state = State(
            cube=cube,
            step_count=step_count,
            key=state.key,
            action_history=action_history,
        )
        reward = self.reward_function(state=next_state)
        solved = is_solved(cube)
        done = (step_count >= self.time_limit) | solved
        next_observation = self._state_to_observation(state=next_state)
        next_timestep = jax.lax.cond(
            done,
            termination,
            transition,
            reward,
            next_observation,
        )
        return next_state, next_timestep

    def get_action_history(self, state: State) -> chex.Array:
        """Slice and return the action history from the state."""
        action_history_index = self.num_scrambles_on_reset + state.step_count
        return state.action_history[:action_history_index]

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specifications of the observation of the `RubiksCube` environment.

        Returns:
            Spec containing all the specifications for all the `Observation` fields:
             - cube: BoundedArray (jnp.int8) of shape (num_faces, cube_size, cube_size).
             - step_count: BoundedArray (jnp.int32) of shape ().
        """
        cube = specs.BoundedArray(
            shape=(len(Face), self.cube_size, self.cube_size),
            dtype=jnp.int8,
            minimum=0,
            maximum=len(Face) - 1,
            name="cube",
        )
        step_count = specs.BoundedArray(
            shape=(),
            dtype=jnp.int32,
            minimum=0,
            maximum=self.time_limit,
            name="step_count",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            cube=cube,
            step_count=step_count,
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec. An action is composed of 3 elements that range in: 6 faces, each
        with cube_size//2 possible depths, and 3 possible directions.

        Returns:
            action_spec: `MultiDiscreteArray` object.
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.array([len(Face), self.cube_size // 2, 3], jnp.int32),
            name="action",
            dtype=jnp.int32,
        )

    def _unflatten_action(self, action: chex.Array) -> chex.Array:
        """Turn a flat action (index into the sequence of all moves) into a tuple:
            - face (0-5). This indicates the face on which the layer will turn.
            - depth (0-cube_size//2). This indicates how many layers down from the face
                the turn will take place.
            - amount (0-2). This indicates the amount of turning (see below).

        Convention:
        - 0 = up face
        - 1 = front face
        - 2 = right face
        - 3 = back face
        - 4 = left face
        - 5 = down face
        All read in reading order when looking directly at a face.

        To look directly at the faces:
        - UP: LEFT face on the left and BACK face pointing up
        - FRONT: LEFT face on the left and UP face pointing up
        - RIGHT: FRONT face on the left and UP face pointing up
        - BACK: RIGHT face on the left and UP face pointing up
        - LEFT: BACK face on the left and UP face pointing up
        - DOWN: LEFT face on the left and FRONT face pointing up

        Turning amounts are when looking directly at a face:
        - 0 = clockwise turn
        - 1 = anticlockwise turn
        - 2 = half turn
        """
        face_and_depth, amount = jnp.divmod(action, len(CubeMovementAmount))
        face, depth = jnp.divmod(face_and_depth, self.cube_size // 2)
        return jnp.stack([face, depth, amount], axis=0)

    def _flatten_action(self, action: chex.Array) -> chex.Array:
        """Inverse of the `_flatten_action` method."""
        face, depth, amount = action
        return (
            face * len(CubeMovementAmount) * (self.cube_size // 2)
            + depth * len(CubeMovementAmount)
            + amount
        )

    def _rotate_cube(self, cube: Cube, flat_action: chex.Array) -> Cube:
        """Apply a flattened action (index into the sequence of all moves) to a cube."""
        moved_cube = jax.lax.switch(flat_action, self.all_moves, cube)
        return moved_cube

    def _scramble_solved_cube(self, flat_actions_in_scramble: chex.Array) -> Cube:
        """Return a scrambled cube according to a given sequence of flat actions."""
        cube = make_solved_cube(cube_size=self.cube_size)
        cube, _ = jax.lax.scan(
            lambda *args: (self._rotate_cube(*args), None),
            cube,
            flat_actions_in_scramble,
        )
        return cube

    def _state_to_observation(self, state: State) -> Observation:
        return Observation(cube=state.cube, step_count=state.step_count)

    def render(self, state: State) -> None:
        """Render frames of the environment for a given state using matplotlib.

        Args:
            state: `State` object corresponding to the new state of the environment.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        self._draw(ax, state)
        self._update_display(fig)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of environment states.

        Args:
            states: sequence of environment states corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=self.figure_size)
        fig.suptitle(self.figure_name)
        plt.tight_layout()
        ax = ax.flatten()
        plt.close(fig)

        def make_frame(state_index: int) -> None:
            state = states[state_index]
            self._draw(ax, state)

        # Create the animation object.
        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=len(states),
            interval=interval,
        )

        # Save the animation as a gif.
        if save_path:
            self._animation.save(save_path)

        return self._animation

    def close(self) -> None:
        """Perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        plt.close(self.figure_name)

    def _get_fig_ax(self) -> Tuple[plt.Figure, List[plt.Axes]]:
        exists = plt.fignum_exists(self.figure_name)
        if exists:
            fig = plt.figure(self.figure_name)
            ax = fig.get_axes()
        else:
            fig, ax = plt.subplots(nrows=3, ncols=2, figsize=self.figure_size)
            fig.suptitle(self.figure_name)
            ax = ax.flatten()
            plt.tight_layout()
            plt.axis("off")
            if not plt.isinteractive():
                fig.show()
        return fig, ax

    def _draw(self, ax: List[plt.Axes], state: State) -> None:
        i = 0
        for face in Face:
            ax[i].clear()
            ax[i].set_title(label=f"{face}")
            ax[i].set_xticks(jnp.arange(-0.5, self.cube_size - 1, 1))
            ax[i].set_yticks(jnp.arange(-0.5, self.cube_size - 1, 1))
            ax[i].tick_params(
                top=False,
                bottom=False,
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
                labeltop=False,
                labelright=False,
            )
            ax[i].imshow(
                state.cube[i],
                cmap=self.sticker_colors_cmap,
                vmin=0,
                vmax=len(Face) - 1,
            )
            ax[i].grid(color="black", linestyle="-", linewidth=2)
            i += 1

    def _update_display(self, fig: plt.Figure) -> None:
        if plt.isinteractive():
            # Required to update render when using Jupyter Notebook.
            fig.canvas.draw()
            if jumanji.environments.is_colab():
                plt.show(self.figure_name)
        else:
            # Required to update render when not using Jupyter Notebook.
            fig.canvas.draw_idle()
            # Block for 0.5 seconds.
            fig.canvas.start_event_loop(0.5)

    def _clear_display(self) -> None:
        if jumanji.environments.is_colab():
            import IPython.display

            IPython.display.clear_output(True)
