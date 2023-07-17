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
from jumanji.environments.routing.boxoban.constants import (
    AGENT,
    BOX,
    EMPTY,
    GRID_SIZE,
    LEVEL_COMPLETE_BONUS,
    MOVES,
    N_BOXES,
    SINGLE_BOX_BONUS,
    STEP_BONUS,
    TARGET,
    TARGET_AGENT,
    TARGET_BOX,
    WALL,
)
from jumanji.environments.routing.boxoban.generator import DeepMindGenerator, Generator
from jumanji.environments.routing.boxoban.types import Observation, State
from jumanji.environments.routing.boxoban.viewer import BoxViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class Boxoban(Environment[State]):
    """A JAX implementation of the 'Boxoban' game from deepmind.

    - observation: `Observation`
        - grid: jax array (uint8) of shape (num_rows, num_cols, 2)
            Array that includes information about the agent, boxes, and
            targets in the game.
        - step_count: jax array (int32) of shape ()
            current number of steps in the episode.

    - action: jax array (int32) of shape ()
        [0,1,2,3] -> [Up, Down, Left, Right].

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
            array indicating the agent and boxes in the level.
        - agent_location: jax array (int32) of shape (2,)
            the agent's current location.
        - step_count: jax array (int32) of shape ()
            current number of steps in the episode.

    ```python
    from jumanji.environments import Boxoban
    env = Boxoban()
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
        time_limit: int = 120,
        viewer: Optional[Viewer] = None,
    ) -> None:
        """
        Instantiates a `Boxoban` environment with a specific generator,
        time limit, and viewer.

        Args:
            generator: `Generator` whose `__call__` instantiates an environment
             instance. Implemented options are [`ToyGenerator`,
             `DeepMindGenerator`].Defaults to `DeepMindGenerator` with
             'difficulty=unfiltered','split=train', 'proportion_of_files=0.01'.
            time_limit: int, max steps for the environment ,defaults to 120.
            viewer: 'Viewer' object, used to render the environment.
            If not provided, defaults to`BoxViewer`.
        """

        self.num_rows = GRID_SIZE
        self.num_cols = GRID_SIZE
        self.shape = (self.num_rows, self.num_cols)
        self.time_limit = time_limit
        self.generator = generator or DeepMindGenerator(
            difficulty="unfiltered",
            split="train",
            proportion_of_files=0.01,
        )
        self._viewer = viewer or BoxViewer(
            name="Boxoban",
            grid_combine=self.grid_combine,
        )

    def __repr__(self) -> str:
        """
        Returns a printable representation of the Boxoban environment.

        Returns:
            str: A string representation of the Boxoban environment.
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
            key: random key used to sample new Boxoban problem.

        Returns:
            state: `State` object corresponding to the new state of the
            environment after a reset.
            timestep: `TimeStep` object corresponding the first timestep
            returned by the environment after a reset.
        """

        generator_key, key = jax.random.split(key)
        fixed_grid, variable_grid = self.generator(generator_key)
        initial_agent_location = self.get_agent_coordinates(variable_grid)

        state = State(
            key=key,
            fixed_grid=fixed_grid,
            variable_grid=variable_grid,
            agent_location=initial_agent_location,
            step_count=jnp.array(0, jnp.int32),
        )

        timestep = restart(
            self._state_to_observation(state),
            extras=self._get_extras(state),
        )

        return state, timestep

    def step(
        self,
        state: State,
        action: chex.Array
        # Maybe this should be chex.Numeric?
    ) -> Tuple[State, TimeStep[Observation]]:
        """
        Executes one timestep of the environment's dynamics.

        Args:
            state: 'State' object representing the current state of the
            environment.
            action: Array (int32) of shape ().
                - 0: move left.
                - 1: move right.
                - 2: move down.
                - 3: move to up.

        Returns:
            state, timestep: next state of the environment and timestep to be
            observed.
        """

        # switch to noop if action will have no impact on variable grid
        action = self.detect_noop_action(state.variable_grid, state.fixed_grid, action)

        next_variable_grid, next_agent_location = self.update_grid_and_agent(
            state,
            action,
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
        done = target_reached | time_limit_exceeded

        reward = jnp.asarray(self.reward(state, next_state), float)

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
        Returns the specifications of the observation of the `Boxoban`
        environment.

        Returns:
            specs.Spec[Observation]: The specifications of the observations.
        """
        grid = specs.BoundedArray(  # What is this grid I am a bit confused
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
        Returns the action specification for the Boxoban environment.
        There are 4 actions: [0,1,2,3] -> [Up, Down, Left, Right].

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

        total_grid = jnp.concatenate(
            (
                jnp.expand_dims(state.variable_grid, axis=-1),
                jnp.expand_dims(state.fixed_grid, axis=-1),
            ),
            axis=-1,
        )

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
            targets
        """
        num_boxes_on_targets = self.count_targets(state)
        total_num_boxes = N_BOXES
        extras = {
            "prop_correct_boxes": num_boxes_on_targets / total_num_boxes,
        }
        return extras

    def get_agent_coordinates(self, grid: chex.Array) -> chex.Array:
        """Extracts the coordinates of the agent from a given grid with the
        assumption there is only one agent in the grid.

        Args:
            grid: Array (uint8) of shape (num_rows, num_cols)

        Returns:
            location: (int32) of shape (2,)
        """

        coordinates = jnp.where(grid == AGENT, size=1)

        # Is this the right way to extract?
        x_coord = jnp.squeeze(coordinates[0])
        y_coord = jnp.squeeze(coordinates[1])

        return jnp.array([x_coord, y_coord])

    def grid_combine(
        self, variable_grid: chex.Array, fixed_grid: chex.Array
    ) -> chex.Array:
        """
        Combines the variable grid and fixed grid into one single grid
        representation of the current Boxoban state.

        Args:
            variable_grid: Array (uint8) of shape (num_rows, num_cols).
            fixed_grid: Array (uint8) of shape (num_rows, num_cols).

        Returns:
            full_grid: Array (uint8) of shape (num_rows, num_cols, 2).
        """

        # This is broken!!!

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

    def count_targets(self, state: State) -> chex.Array:
        """
        Calculates the number of boxes on targets.

        Args:
            state: `State` object representing the current state of the
            environment.

        Returns:
            n_targets: Array (int32) of shape () specifying the number of boxes on
            targets.
        """

        mask_box = state.variable_grid == BOX
        mask_target = state.fixed_grid == TARGET

        num_boxes_on_targets = jnp.sum(mask_box & mask_target)

        return num_boxes_on_targets

    def reward(self, state: State, next_state: State) -> chex.Array:
        """
        Implements the reward function in the Boxoban environment.

        Args:
            state: `State` object The current state of the environment.
            next_state:  `State` object The next state of the environment.

        Returns:
            reward: Array (float32) of shape () specifying the reward received
            at transition
        """

        num_box_target = self.count_targets(state)
        next_num_box_target = self.count_targets(next_state)

        level_completed = next_num_box_target == N_BOXES

        return (
            SINGLE_BOX_BONUS * (next_num_box_target - num_box_target)
            + LEVEL_COMPLETE_BONUS * level_completed
            + STEP_BONUS
        )

    def level_complete(self, state: State) -> chex.Array:
        """
        Checks if the sokoban level is complete.

        Args:
            state: `State` object representing the current state of the environment.

        Returns:
            complete: Boolean indicating whether the level is complete
            or not.
        """
        return self.count_targets(state) == N_BOXES

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
    ) -> chex.Array:
        """
        Masks actions to -1 that have no effect on the variable grid.
        Determines if there is space in the destination square or if
        there is a box in the destination square, it determines if the box
        destination square is free.

        Args:
            variable_grid: Array (uint8) shape (num_rows, num_cols).
            fixed_grid Array (uint8) shape (num_rows, num_cols) .
            action: Array (int32) shape () The action to check.

        Returns:
            updated_action: Array (int32) shape () The updated action after
            detecting noop action.
        """

        new_location = self.get_agent_coordinates(variable_grid) + MOVES[action]

        valid_destination = self.check_space(
            fixed_grid, new_location, WALL
        ) | ~self.in_grid(new_location)

        updated_action = jax.lax.select(
            valid_destination,
            -jnp.ones(shape=(), dtype=jnp.int32),
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
            -jnp.ones(shape=(), dtype=jnp.int32),
            jax.lax.select(
                self.check_space(
                    fixed_grid,
                    new_location + MOVES[action].squeeze(),
                    WALL,
                ),
                -jnp.ones(shape=(), dtype=jnp.int32),
                action,
            ),
        )

    # still think a better name can be used or we can remove the whole thigng
    def update_grid_and_agent(
        self,
        state: State,
        action: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Moves the agent in the grid if an action other than noop is given.

        Args:
            state: 'State' object representing the current state of the
            environment.
            action: Array (int32) shape () representing the action to take.

        Returns:
            next_grid: Array (uint8) shape (num_rows, num_cols) The updated
            grid.
            next_agent_location: Array (int32) shape (2,) updated Agent
            location
        """

        next_grid, next_agent_location = jax.lax.cond(
            jnp.all(action == -1),
            lambda: (state.variable_grid, state.agent_location),
            lambda: self.move_agent(state.variable_grid, action, state.agent_location),
        )

        return next_grid, next_agent_location

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
        next_variable_grid = jax.lax.cond(
            self.check_space(variable_grid, next_location, BOX),
            lambda: next_variable_grid.at[tuple(next_location)]
            .set(AGENT)
            .at[tuple(box_location)]
            .set(BOX),
            lambda: next_variable_grid.at[tuple(next_location)].set(AGENT),
        )

        return next_variable_grid, next_location

    def render(self, state: State) -> None:
        """
        Renders the current state of Boxoban.

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
        Creates an animated gif of the Boxoban environment based on the
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
