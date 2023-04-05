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
import matplotlib.animation
from numpy.typing import NDArray

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.logic.rubiks_cube.constants import (
    DEFAULT_STICKER_COLORS,
    Face,
)
from jumanji.environments.logic.rubiks_cube.generator import (
    Generator,
    ScramblingGenerator,
)
from jumanji.environments.logic.rubiks_cube.reward import RewardFn, SparseRewardFn
from jumanji.environments.logic.rubiks_cube.types import Observation, State
from jumanji.environments.logic.rubiks_cube.utils import (
    flatten_action,
    is_solved,
    rotate_cube,
)
from jumanji.environments.logic.rubiks_cube.viewer import RubiksCubeViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


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
        generator: Optional[Generator] = None,
        time_limit: int = 200,
        reward_fn: Optional[RewardFn] = None,
        viewer: Optional[Viewer[State]] = None,
    ):
        """Instantiate a `RubiksCube` environment.

        Args:
            generator: `Generator` used to generate problem instances on environment reset.
                Implemented options are [`ScramblingGenerator`]. Defaults to `ScramblingGenerator`,
                with 100 scrambles on reset.
                The generator will contain an attribute `cube_size`, corresponding to the number of
                cubies to an edge, and defaulting to 3.
            time_limit: the number of steps allowed before an episode terminates. Defaults to 200.
            reward_fn: `RewardFn` whose `__call__` method computes the reward given the new state.
                Implemented options are [`SparseRewardFn`]. Defaults to `SparseRewardFn`, giving a
                reward of 1.0 if the cube is solved or otherwise 0.0.
            viewer: `Viewer` to support rendering and animation methods.
                Implemented options are [`RubiksCubeViewer`]. Defaults to `RubiksCubeViewer`.
        """
        if time_limit <= 0:
            raise ValueError(
                f"The time_limit must be positive, but received time_limit={time_limit}"
            )
        self.time_limit = time_limit
        self.reward_function = reward_fn or SparseRewardFn()
        self.generator = generator or ScramblingGenerator(
            cube_size=3,
            num_scrambles_on_reset=100,
        )
        self._viewer = viewer or RubiksCubeViewer(
            sticker_colors=DEFAULT_STICKER_COLORS, cube_size=self.generator.cube_size
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment.

        Args:
            key: needed for scramble.

        Returns:
            state: `State` corresponding to the new state of the environment.
            timestep: `TimeStep` corresponding to the first timestep returned by the
                environment.
        """
        state = self.generator(key)
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
        flattened_action = flatten_action(
            unflattened_action=action, cube_size=self.generator.cube_size
        )
        cube = rotate_cube(
            cube=state.cube,
            flattened_action=flattened_action,
        )
        step_count = state.step_count + 1
        next_state = State(
            cube=cube,
            step_count=step_count,
            key=state.key,
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

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specifications of the observation of the `RubiksCube` environment.

        Returns:
            Spec containing all the specifications for all the `Observation` fields:
             - cube: BoundedArray (jnp.int8) of shape (num_faces, cube_size, cube_size).
             - step_count: BoundedArray (jnp.int32) of shape ().
        """
        cube = specs.BoundedArray(
            shape=(len(Face), self.generator.cube_size, self.generator.cube_size),
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
            num_values=jnp.array(
                [len(Face), self.generator.cube_size // 2, 3], jnp.int32
            ),
            name="action",
            dtype=jnp.int32,
        )

    def _state_to_observation(self, state: State) -> Observation:
        return Observation(cube=state.cube, step_count=state.step_count)

    def render(self, state: State) -> Optional[NDArray]:
        """Renders the current state of the cube.

        Args:
            state: the current state to be rendered.
        """
        return self._viewer.render(state=state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Creates an animated gif of the cube based on the sequence of states.

        Args:
            states: a list of `State` objects representing the sequence of game states.
            interval: the delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

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
