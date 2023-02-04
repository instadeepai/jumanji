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
import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
from chex import PRNGKey
from jax import lax
from jax import numpy as jnp

import jumanji.environments
from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.logic.rubiks_cube.constants import (
    DEFAULT_CUBE_SIZE,
    DEFAULT_STICKER_COLOURS,
    CubeMovementAmount,
    Face,
)
from jumanji.environments.logic.rubiks_cube.reward_functions import (
    RewardFunction,
    SparseRewardFunction,
)
from jumanji.environments.logic.rubiks_cube.specs import ObservationSpec
from jumanji.environments.logic.rubiks_cube.types import Cube, Observation, State
from jumanji.environments.logic.rubiks_cube.utils import (
    generate_all_moves,
    is_solved,
    make_solved_cube,
)
from jumanji.types import Action, TimeStep, restart, termination, transition


def state_to_observation(state: State) -> Observation:
    return Observation(cube=state.cube, step_count=state.step_count)


class RubiksCube(Environment[State]):
    """A JAX implementation of the Rubik's Cube.

    - observation: Observation
        - cube: jax array (int8) of shape (6, cube_size, cube_size):
            each cell contains the index of the corresponding colour of the sticker in the scramble.
        - step_count: jax array (int32):
            specifies how many timesteps have elapsed since environment reset

    - action:
        multi discrete array containing the move to perform (face, depth, and direction)

    - reward: jax array (float):
        by default, 1 if cube is solved or otherwise 0

    - episode termination:
        if either the cube is solved or a horizon is reached

    - state: State
        - cube: as in observation.
        - step_count: as in observation.
        - key: jax array (int32) of shape (2) used for seeding the sampling for scrambling on reset.
        - action_history: jax array (int16) of shape (max_num_scrambles, 3):
            indicates the entire history of applied moves
            (including those taken on scrambling the cube in the environment reset).
            max_num_scrambles = num_scrambles_on_reset + step_limit.
            This is useful for debugging purposes, providing a method to solve the cube from any
            position without relying on the agent, by just inverting the action history.
            The first axis indexes over the length of the sequence
            The second axis indexes over the component of the action (face, depth, amount).
            The number of scrambles applied for each state is given by
            env.num_scrambles_on_reset + state.step_count
    """

    def __init__(
        self,
        cube_size: int = DEFAULT_CUBE_SIZE,
        step_limit: int = 200,
        reward_function_type: str = "sparse",
        num_scrambles_on_reset: int = 100,
        sticker_colours: Optional[list] = None,
    ):
        if cube_size <= 1:
            raise ValueError(
                f"Cannot meaningfully construct a cube smaller than 2x2x2, "
                f"but received cube_size={cube_size}"
            )
        if step_limit <= 0:
            raise ValueError(
                f"The step_limit must be positive, but received step_limit={step_limit}"
            )
        if num_scrambles_on_reset < 0:
            raise ValueError(
                f"The num_scrambles_on_reset must be non-negative, "
                f"but received num_scrambles_on_reset={num_scrambles_on_reset}"
            )
        self.cube_size = cube_size
        self.step_limit = step_limit
        self.reward_function = self.create_reward_function(
            reward_function_type=reward_function_type
        )
        self.num_scrambles_on_reset = num_scrambles_on_reset
        self.sticker_colours_cmap = matplotlib.colors.ListedColormap(
            sticker_colours if sticker_colours else DEFAULT_STICKER_COLOURS
        )
        self.num_actions = len(Face) * (cube_size // 2) * len(CubeMovementAmount)
        self.all_moves = generate_all_moves(cube_size=cube_size)

        self.figure_name = f"{cube_size}x{cube_size}x{cube_size} Rubik's Cube"
        self.figure_size = (6.0, 6.0)

    @classmethod
    def create_reward_function(cls, reward_function_type: str) -> RewardFunction:
        if reward_function_type == "sparse":
            return SparseRewardFunction()
        else:
            raise ValueError(
                f"Unexpected value for reward_function_type, got {reward_function_type}. "
                f"Possible values: 'sparse'"
            )

    def _unflatten_action(self, action: chex.Array) -> chex.Array:
        """Turn a flat action (index into the sequence of all moves) into a tuple:
            - face (0-5). This indicates the face on which the layer will turn.
            - depth (0-cube_size//2). This indicates how many layers down from the face
                the turn will take place.
            - amount (0-2). This indicates the amount of turning (see below).

        Convention:
        0 = up face
        1 = front face
        2 = right face
        3 = back face
        4 = left face
        5 = down face
        All read in reading order when looking directly at face

        To look directly at the faces:
        UP: LEFT face on the left and BACK face pointing up
        FRONT: LEFT face on the left and UP face pointing up
        RIGHT: FRONT face on the left and UP face pointing up
        BACK: RIGHT face on the left and UP face pointing up
        LEFT: BACK face on the left and UP face pointing up
        DOWN: LEFT face on the left and FRONT face pointing up

        Turning amounts are when looking directly at a face:
        0 = clockwise turn
        1 = anticlockwise turn
        2 = half turn
        """
        face_and_depth, amount = jnp.divmod(action, len(CubeMovementAmount))
        face, depth = jnp.divmod(face_and_depth, self.cube_size // 2)
        return jnp.stack([face, depth, amount], axis=0)

    def _flatten_action(self, action: chex.Array) -> chex.Array:
        """Inverse of the _flatten_action method"""
        face, depth, amount = action
        return (
            face * len(CubeMovementAmount) * (self.cube_size // 2)
            + depth * len(CubeMovementAmount)
            + amount
        )

    def _rotate_cube(self, cube: Cube, flat_action: chex.Array) -> Cube:
        """Apply a flattened action (index into the sequence of all moves) to a cube"""
        moved_cube = lax.switch(flat_action, self.all_moves, cube)
        return moved_cube

    def _scramble_solved_cube(self, flat_actions_in_scramble: chex.Array) -> Cube:
        """Return a scrambled cube according to a given sequence of flat actions"""
        cube = make_solved_cube(cube_size=self.cube_size)
        cube, _ = lax.scan(
            lambda *args: (self._rotate_cube(*args), None),
            cube,
            flat_actions_in_scramble,
        )
        return cube

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment.

        Args:
            key: needed for scramble.

        Returns:
            state: State corresponding to the new state of the environment,
            timestep: TimeStep corresponding to the first timestep returned by the
                environment.
        """
        flat_actions_in_scramble = jax.random.randint(
            key,
            minval=0,
            maxval=self.num_actions,
            shape=(self.num_scrambles_on_reset,),
            dtype=jnp.int16,
        )
        cube = self._scramble_solved_cube(
            flat_actions_in_scramble=flat_actions_in_scramble
        )
        action_history = jnp.zeros(
            shape=(self.num_scrambles_on_reset + self.step_limit, 3), dtype=jnp.int16
        )
        action_history = action_history.at[: self.num_scrambles_on_reset].set(
            self._unflatten_action(flat_actions_in_scramble).transpose()
        )
        step_count = jnp.int32(0)
        state = State(
            cube=cube,
            step_count=step_count,
            key=key,
            action_history=action_history,
        )
        observation = state_to_observation(state=state)
        timestep = restart(observation=observation)
        return state, timestep

    def step(self, state: State, action: Action) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the face to move, depth of the move, and the amount
                to move by.

        Returns:
            next_state: State corresponding to the next state of the environment,
            next_timestep: TimeStep corresponding to the timestep returned by the environment.
        """
        cube = state.cube
        cube = self._rotate_cube(cube=cube, flat_action=self._flatten_action(action))
        step_count = state.step_count + 1
        action_history = state.action_history
        action_history.at[self.num_scrambles_on_reset + state.step_count].set(action)
        next_state = State(
            cube=cube,
            step_count=step_count,
            key=state.key,
            action_history=action_history,
        )
        reward = self.reward_function(state=next_state)
        solved = is_solved(cube)
        done = (step_count >= self.step_limit) | solved
        next_observation = state_to_observation(state=next_state)
        next_timestep = lax.cond(
            done,
            termination,
            transition,
            reward,
            next_observation,
        )
        return next_state, next_timestep

    def get_action_history(self, state: State) -> chex.Array:
        """Slice and return the action history from the state"""
        action_history_index = self.num_scrambles_on_reset + state.step_count
        return state.action_history[:action_history_index]

    def observation_spec(self) -> ObservationSpec:
        """Returns the observation spec containing the cube and step count.

        Returns:
            observation_spec: ObservationSpec tree of cube and step_count spec.
        """
        return ObservationSpec(
            cube=specs.BoundedArray(
                shape=(len(Face), self.cube_size, self.cube_size),
                dtype=jnp.int8,
                minimum=0,
                maximum=len(Face) - 1,
                name="cube",
            ),
            step_count=specs.BoundedArray(
                shape=(),
                dtype=jnp.int32,
                minimum=0,
                maximum=self.step_limit,
                name="step_count",
            ),
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec.
        6 faces, each with cube_size//2 possible depths, and 3 possible directions

        Returns:
            action_spec: specs.MultiDiscreteArray object
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.array(
                [len(Face), self.cube_size // 2, len(CubeMovementAmount)]
            ),
            name="action",
            dtype=jnp.int16,
        )

    def render(self, state: State) -> None:
        """Render frames of the environment for a given state using matplotlib.

        Args:
            state: State object corresponding to the new state of the environment.

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
                cmap=self.sticker_colours_cmap,
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

    def animation(
        self,
        states: Sequence[State],
        interval: int = 200,
        blit: bool = False,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of environment states.

        Args:
            states: sequence of environment states corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            blit: whether to use blitting, which optimises the animation by only re-drawing
                pieces of the plot that have changed. Defaults to False.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=self.figure_size)
        fig.suptitle(self.figure_name)
        plt.tight_layout()
        ax = ax.flatten()
        plt.close(fig)

        def animate(state_index: int) -> None:
            state = states[state_index]
            self._draw(ax, state)

        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            animate,
            frames=len(states),
            blit=blit,
            interval=interval,
        )
        return self._animation
