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
import matplotlib.animation

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.sokoban.constants import (
    AGENT,
    BOX,
    EMPTY,
    GRID_SIZE,
    MOVES,
    N_BOXES,
    NOOP,
    TARGET,
    TARGET_AGENT,
    TARGET_BOX,
    WALL,
)
from jumanji.environments.routing.sokoban.generator import (
    Generator,
    HuggingFaceDeepMindGenerator,
)
from jumanji.environments.routing.sokoban.reward import DenseReward, RewardFn
from jumanji.environments.routing.sokoban.types import Observation, State
from jumanji.environments.routing.sokoban.viewer import BoxViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class Sokoban(Environment[State]):
    """A JAX implementation of the 'Sokoban' game from deepmind.

    - observation: `Observation`
        - grid: jax array (uint8) of shape (num_rows, num_cols, 2)
            Array that includes information about the agent, boxes, and
            targets in the game.
        - step_count: jax array (int32) of shape ()
            current number of steps in the episode.

    - action: jax array (int32) of shape ()
        [0,1,2,3] -> [Up, Right, Down, Left].

    - reward: jax array (float) of shape ()
        A reward of 1.0 is given for each box placed on a target and -1
        when removed from a target and -0.1 for each timestep.
        10 is awarded when all boxes are on targets.

    - episode termination:
        - if the time limit is reached.
        - if all boxes are on targets.

    - state: `State`
        - key: jax array (uint32) of shape (2,) used for auto-reset
        - fixed_grid: jax array (uint8) of shape (num_rows, num_cols)
            array indicating the walls and targets in the level.
        - variable_grid: jax array (uint8) of shape (num_rows, num_cols)
            array indicating the current location of the agent and boxes.
        - agent_location: jax array (int32) of shape (2,)
            the agent's current location.
        - step_count: jax array (int32) of shape ()
            current number of steps in the episode.

    ```python
    from jumanji.environments import Sokoban
    from jumanji.environments.routing.sokoban.generator import
    HuggingFaceDeepMindGenerator,

    env_train = Sokoban(
        generator=HuggingFaceDeepMindGenerator(
            dataset_name="unfiltered-train",
            proportion_of_files=1,
        )
    )

    env_test = Sokoban(
        generator=HuggingFaceDeepMindGenerator(
            dataset_name="unfiltered-test",
            proportion_of_files=1,
        )
    )

    # Train...

    ```
    key_train = jax.random.PRNGKey(0)
    state, timestep = jax.jit(env_train.reset)(key_train)
    env_train.render(state)
    action = env_train.action_spec().generate_value()
    state, timestep = jax.jit(env_train.step)(state, action)
    env_train.render(state)
    ```
    """

    def __init__(
        self,
        generator: Optional[Generator] = None,
        reward_fn: Optional[RewardFn] = None,
        viewer: Optional[Viewer] = None,
        time_limit: int = 120,
    ) -> None:
        """
        Instantiates a `Sokoban` environment with a specific generator,
        time limit, and viewer.

        Args:
            generator: `Generator` whose `__call__` instantiates an environment
             instance (an initial state). Implemented options are [`ToyGenerator`,
             `DeepMindGenerator`, and `HuggingFaceDeepMindGenerator`].
             Defaults to `HuggingFaceDeepMindGenerator` with
             `dataset_name="unfiltered-train", proportion_of_files=1`.
            time_limit: int, max steps for the environment, defaults to 120.
            viewer: 'Viewer' object, used to render the environment.
            If not provided, defaults to`BoxViewer`.
        """

        self.num_rows = GRID_SIZE
        self.num_cols = GRID_SIZE
        self.shape = (self.num_rows, self.num_cols)
        self.time_limit = time_limit

        self.generator = generator or HuggingFaceDeepMindGenerator(
            "unfiltered-train",
            proportion_of_files=1,
        )

        self._viewer = viewer or BoxViewer(
            name="Sokoban",
            grid_combine=self.grid_combine,
        )
        self.reward_fn = reward_fn or DenseReward()

    def __repr__(self) -> str:
        """
        Returns a printable representation of the Sokoban environment.

        Returns:
            str: A string representation of the Sokoban environment.
        """
        return "\n".join(
            [
                "Bokoban environment:",
                f" - num_rows: {self.num_rows}",
                f" - num_cols: {self.num_cols}",
                f" - time_limit: {self.time_limit}",
                f" - generator: {self.generator}",
            ]
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """
        Resets the environment by calling the instance generator for a
        new instance.

        Args:
            key: random key used to sample new Sokoban problem.

        Returns:
            state: `State` object corresponding to the new state of the
            environment after a reset.
            timestep: `TimeStep` object corresponding the first timestep
            returned by the environment after a reset.
        """

        generator_key, key = jax.random.split(key)

        state = self.generator(generator_key)

        timestep = restart(
            self._state_to_observation(state),
            extras=self._get_extras(state),
        )

        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        """
        Executes one timestep of the environment's dynamics.

        Args:
            state: 'State' object representing the current state of the
            environment.
            action: Array (int32) of shape ().
                - 0: move up.
                - 1: move down.
                - 2: move left.
                - 3: move right.

        Returns:
            state, timestep: next state of the environment and timestep to be
            observed.
        """

        # switch to noop if action will have no impact on variable grid
        action = self.detect_noop_action(
            state.variable_grid, state.fixed_grid, action, state.agent_location
        )

        next_variable_grid, next_agent_location = jax.lax.cond(
            jnp.all(action == NOOP),
            lambda: (state.variable_grid, state.agent_location),
            lambda: self.move_agent(state.variable_grid, action, state.agent_location),
        )

        next_state = State(
            key=state.key,
            fixed_grid=state.fixed_grid,
            variable_grid=next_variable_grid,
            agent_location=next_agent_location,
            step_count=state.step_count + 1,
        )

        target_reached = self.level_complete(next_state)
        time_limit_exceeded = next_state.step_count >= self.time_limit

        done = jnp.logical_or(target_reached, time_limit_exceeded)

        reward = jnp.asarray(self.reward_fn(state, action, next_state), float)

        observation = self._state_to_observation(next_state)

        extras = self._get_extras(next_state)

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

        return next_state, timestep

    def observation_spec(self) -> specs.Spec[Observation]:
        """
        Returns the specifications of the observation of the `Sokoban`
        environment.

        Returns:
            specs.Spec[Observation]: The specifications of the observations.
        """
        grid = specs.BoundedArray(
            shape=(self.num_rows, self.num_cols, 2),
            dtype=jnp.uint8,
            minimum=0,
            maximum=4,
            name="grid",
        )
        step_count = specs.Array((), jnp.int32, "step_count")
        return specs.Spec(
            Observation,
            "ObservationSpec",
            grid=grid,
            step_count=step_count,
        )

    def action_spec(self) -> specs.DiscreteArray:
        """
        Returns the action specification for the Sokoban environment.
        There are 4 actions: [0,1,2,3] -> [Up, Right, Down, Left].

        Returns:
            specs.DiscreteArray: Discrete action specifications.
        """
        return specs.DiscreteArray(4, name="action", dtype=jnp.int32)

    def _state_to_observation(self, state: State) -> Observation:
        """Maps an environment state to an observation.

        Args:
            state: `State` object containing the dynamics of the environment.

        Returns:
            The observation derived from the state.
        """

        total_grid = jnp.stack([state.variable_grid, state.fixed_grid], axis=-1)

        return Observation(
            grid=total_grid,
            step_count=state.step_count,
        )

    def _get_extras(self, state: State) -> Dict:
        """
        Computes extras metrics to be returned within the timestep.

        Args:
            state: 'State' object representing the current state of the
            environment.

        Returns:
            extras: Dict object containing current proportion of boxes on
            targets and whether the problem is solved.
        """
        num_boxes_on_targets = self.reward_fn.count_targets(state)
        total_num_boxes = N_BOXES
        extras = {
            "prop_correct_boxes": num_boxes_on_targets / total_num_boxes,
            "solved": num_boxes_on_targets == 4,
        }
        return extras

    def grid_combine(
        self, variable_grid: chex.Array, fixed_grid: chex.Array
    ) -> chex.Array:
        """
        Combines the variable grid and fixed grid into one single grid
        representation of the current Sokoban state required for visual
        representation of the Sokoban state. Takes care of two possible
        overlaps of fixed and variable entries (an agent on a target or a box
        on a target), introducing two additional encodings.

        Args:
            variable_grid: Array (uint8) of shape (num_rows, num_cols).
            fixed_grid: Array (uint8) of shape (num_rows, num_cols).

        Returns:
            full_grid: Array (uint8) of shape (num_rows, num_cols, 2).
        """

        mask_target_agent = jnp.logical_and(
            fixed_grid == TARGET,
            variable_grid == AGENT,
        )

        mask_target_box = jnp.logical_and(
            fixed_grid == TARGET,
            variable_grid == BOX,
        )

        single_grid = jnp.where(
            mask_target_agent,
            TARGET_AGENT,
            jnp.where(
                mask_target_box,
                TARGET_BOX,
                jnp.maximum(variable_grid, fixed_grid),
            ),
        ).astype(jnp.uint8)

        return single_grid

    def level_complete(self, state: State) -> chex.Array:
        """
        Checks if the sokoban level is complete.

        Args:
            state: `State` object representing the current state of the environment.

        Returns:
            complete: Boolean indicating whether the level is complete
            or not.
        """
        return self.reward_fn.count_targets(state) == N_BOXES

    def check_space(
        self,
        grid: chex.Array,
        location: chex.Array,
        value: int,
    ) -> chex.Array:
        """
        Checks if a specific location in the grid contains a given value.

        Args:
            grid: Array (uint8) shape (num_rows, num_cols) The grid to check.
            location: Tuple size 2 of Array (int32) shape () containing the x
            and y coodinate of the location to check in the grid.
            value: int The value to look for.

        Returns:
            present: Array (bool) shape () indicating whether the location
            in the grid contains the given value or not.
        """

        return grid[tuple(location)] == value

    def in_grid(self, coordinates: chex.Array) -> chex.Array:
        """
        Checks if given coordinates are within the grid size.

        Args:
            coordinates: Array (uint8) shape (num_rows, num_cols) The
            coordinates to check.
        Returns:
            in_grid: Array (bool) shape () Boolean indicating whether the
            coordinates are within the grid.
        """
        return jnp.all((0 <= coordinates) & (coordinates < GRID_SIZE))

    def detect_noop_action(
        self,
        variable_grid: chex.Array,
        fixed_grid: chex.Array,
        action: chex.Array,
        agent_location: chex.Array,
    ) -> chex.Array:
        """
        Masks actions to -1 that have no effect on the variable grid.
        Determines if there is space in the destination square or if
        there is a box in the destination square, it determines if the box
        destination square is valid.

        Args:
            variable_grid: Array (uint8) shape (num_rows, num_cols).
            fixed_grid Array (uint8) shape (num_rows, num_cols) .
            action: Array (int32) shape () The action to check.

        Returns:
            updated_action: Array (int32) shape () The updated action after
            detecting noop action.
        """

        new_location = agent_location + MOVES[action].squeeze()

        valid_destination = self.check_space(
            fixed_grid, new_location, WALL
        ) | ~self.in_grid(new_location)

        updated_action = jax.lax.select(
            valid_destination,
            jnp.full(shape=(), fill_value=NOOP, dtype=jnp.int32),
            jax.lax.select(
                self.check_space(variable_grid, new_location, BOX),
                self.update_box_push_action(
                    fixed_grid,
                    variable_grid,
                    new_location,
                    action,
                ),
                action,
            ),
        )

        return updated_action

    def update_box_push_action(
        self,
        fixed_grid: chex.Array,
        variable_grid: chex.Array,
        new_location: chex.Array,
        action: chex.Array,
    ) -> chex.Array:
        """
        Masks actions to -1 if pushing the box is not a valid move. If it
        would be pushed out of the grid or the resulting square
        is either a wall or another box.

        Args:
            fixed_grid: Array (uint8) shape (num_rows, num_cols) The fixed grid.
            variable_grid: Array (uint8) shape (num_rows, num_cols) The
            variable grid.
            new_location: Array (int32) shape (2,) The new location of the agent.
            action: Array (int32) shape () The action to be executed.

        Returns:
            updated_action: Array (int32) shape () The updated action after
            checking if pushing the box is a valid move.
        """

        return jax.lax.select(
            self.check_space(
                variable_grid,
                new_location + MOVES[action].squeeze(),
                BOX,
            )
            | ~self.in_grid(new_location + MOVES[action].squeeze()),
            jnp.full(shape=(), fill_value=NOOP, dtype=jnp.int32),
            jax.lax.select(
                self.check_space(
                    fixed_grid,
                    new_location + MOVES[action].squeeze(),
                    WALL,
                ),
                jnp.full(shape=(), fill_value=NOOP, dtype=jnp.int32),
                action,
            ),
        )

    def move_agent(
        self,
        variable_grid: chex.Array,
        action: chex.Array,
        current_location: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Executes the movement of the agent specified by the action and
        executes the movement of a box if present at the destination.

        Args:
            variable_grid: Array (uint8) shape (num_rows, num_cols)
            action: Array (int32) shape () The action to take.
            current_location: Array (int32) shape (2,)

        Returns:
            next_variable_grid: Array (uint8) shape (num_rows, num_cols)
            next_location: Array (int32) shape (2,)
        """

        next_location = current_location + MOVES[action]
        box_location = next_location + MOVES[action]

        # remove agent from current location
        next_variable_grid = variable_grid.at[tuple(current_location)].set(EMPTY)

        # either move agent or move agent and box

        next_variable_grid = jax.lax.select(
            self.check_space(variable_grid, next_location, BOX),
            next_variable_grid.at[tuple(next_location)]
            .set(AGENT)
            .at[tuple(box_location)]
            .set(BOX),
            next_variable_grid.at[tuple(next_location)].set(AGENT),
        )

        return next_variable_grid, next_location

    def render(self, state: State) -> None:
        """
        Renders the current state of Sokoban.

        Args:
            state: 'State' object , the current state to be rendered.
        """

        self._viewer.render(state=state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """
        Creates an animated gif of the Sokoban environment based on the
        sequence of states.

        Args:
            states: Sequence of 'State' object
            interval: int, The interval between frames in the animation.
            Defaults to 200.
            save_path: str The path where to save the animation. If not
            provided, the animation is not saved.

        Returns:
            animation: 'matplotlib.animation.FuncAnimation'.
        """
        return self._viewer.animate(states, interval, save_path)
