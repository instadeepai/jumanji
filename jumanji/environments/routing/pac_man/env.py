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

from typing import Any, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib.animation
from chex import PRNGKey

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.pac_man.constants import DEFAULT_MAZE, MOVES
from jumanji.environments.routing.pac_man.generator import AsciiGenerator, Generator
from jumanji.environments.routing.pac_man.types import Observation, Position, State
from jumanji.environments.routing.pac_man.utils import (
    check_ghost_collisions,
    ghost_move,
    player_step,
)
from jumanji.environments.routing.pac_man.viewer import PacManViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class PacMan(Environment[State]):
    """A JAX implementation of the 'PacMan' game where a single agent must navigate a
    maze to collect pellets and avoid 4 heuristic agents. The game takes place on a 31x28
    grid where the player can move in 4 directions (left, right, up, down) and collect
    pellets to gain points. The goal is to collect all of the pellets on the board without
    colliding with one of the heuristic agents. Using the AsciiGenerator the environment
    will always generate the same maze as long as the same Ascii diagram is in use.

    - observation: `Observation`
        - player_locations: current 2D position of agent.
        - grid: jax array (int) of the ingame maze with walls.
        - ghost_locations: jax array (int) of ghost positions.
        - power_up_locations: jax array (int) of power-pellet locations
        - pellet_locations: jax array (int) of pellets.
        - action_mask: jax array (bool) defining current actions.
        - score: (int32) of total points aquired.

    - action: jax array (int) of shape () specifiying which action to take [0,1,2,3,4]
        corresponding to [up, right, down, left, no-op. If there is an invalid action
        taken, i.e. there is a wall blocking the action, then no action (no-op) is taken.

    - reward: jax array (float32) of shape (): 10 per pellet collected, 20 for a power pellet
        and 200 for each unique ghost eaten.

    - episode termination (if any):
        - agent has collected all pellets.
        - agent killed by ghost.
        - timer has elapsed.

    - state: State:
        - key: jax array (uint32) of shape(2,).
        - grid: jax array (int)) of shape (31,28) of the ingame maze with walls.
        - pellets: int tracking the number of pellets.
        - frightened_state_time: jax array (int) of shape ()
            tracks number of steps for the scatter state.
        - pellet_locations: jax array (int) of pellets of shape (316,2).
        - power_up_locations: jax array (int) of power-pellet locations of shape (4,2).
        - player_locations: current 2D position of agent.
        - ghost_locations: jax array (int) of ghost positions of shape (4,2).
        - initial_player_locations: starting 2D position of agent.
        - initial_ghost_positions: jax array (int) of ghost positions of shape (4,2).
        - ghost_init_targets: jax array (int) of ghost positions.
            used to direct ghosts on respawn.
        - old_ghost_locations: jax array (int) of shape (4,2) of ghost positions from last step.
            used to prevent ghost backtracking.
        - ghost_init_steps: jax array (int) of shape (4,2) number of initial ghost steps.
            used to determine per ghost initialisation.
        - ghost_actions: jax array (int) of shape (4,).
        - last_direction: int tracking the last direction of the player.
        - dead: bool used to track player death.
        - visited_index: jax array (int) of visited locations of shape (320,2).
            used to prevent repeated pellet points.
        - ghost_starts: jax array (int) of shape (4,2)
            used to reset ghost positions if eaten
        - scatter_targets: jax array (int) of shape (4,2)
            target locations for ghosts when scatter behavior is active.
        - step_count: (int32) of total steps taken from reset till current timestep.
        - ghost_eaten: jax array (bool)of shape (4,) tracking if ghost has been eaten before.
        - score: (int32) of total points aquired.



    ```python
    from jumanji.environments import pac_man
    env = PacMan()
    key = jax.random.PRNGKey(0)
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
        viewer: Optional[Viewer[State]] = None,
        time_limit: Optional[int] = None,
    ) -> None:
        """Instantiates a `PacMan` environment.

        Args:
            generator: `Generator` whose `__call__` instantiates an environment instance.
                Implemented options are [`AsciiGenerator`].
            time_limit: the time_limit of an episode, i.e. the maximum number of environment steps
                before the episode terminates. By default, set to 1000.
            viewer: `Viewer` used for rendering. Defaults to `PacManViewer`.
        """

        self.generator = generator or AsciiGenerator(DEFAULT_MAZE)
        self.x_size = self.generator.x_size
        self.y_size = self.generator.y_size
        self.pellet_spaces = self.generator.pellet_spaces
        self._viewer = viewer or PacManViewer("Pacman", render_mode="human")
        self.time_limit = 1000 or time_limit

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specifications of the observation of the `PacMan` environment.

        Returns:
            Spec containing all the specifications for all the `Observation` fields:
            - player_locations: tree of BoundedArray (int32) of shape ().
            - grid: BoundedArray (int)) of the ingame maze with walls.
            - ghost_locations: jax array (int) of ghost positions.
            - power_up_locations: jax array (int) of power-pellet locations
            - pellet_locations: jax array (int) of pellet locations.
            - action_mask: jax array (bool) defining current actions.
            - frightened_state_time: int counting time remaining in
                scatter mode.
            - score: (int) of total score obtained by player.
        """
        player_locations = specs.Spec(
            Position,
            "PositionSpec",
            y=specs.BoundedArray((), jnp.int32, 0, self.x_size - 1, "y_coordinate"),
            x=specs.BoundedArray((), jnp.int32, 0, self.y_size - 1, "x_coordinate"),
        )
        grid = specs.BoundedArray(
            shape=(self.x_size, self.y_size),
            dtype=jnp.int32,
            name="grid",
            minimum=0,
            maximum=1,
        )
        ghost_locations = specs.Array(
            shape=(4, 2),
            dtype=jnp.int32,
            name="ghost_locations",
        )
        power_up_locations = specs.Array(
            shape=(4, 2),
            dtype=jnp.int32,
            name="power_up_locations",
        )
        pellet_locations = specs.Array(
            shape=(self.pellet_spaces.shape),
            dtype=jnp.int32,
            name="pellet_locations",
        )
        action_mask = specs.BoundedArray(
            shape=(5,),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask",
        )

        frightened_state_time = specs.Array((), jnp.int32, "frightened_state_time")
        score = specs.Array((), jnp.int32, "frightened_state_time")

        return specs.Spec(
            Observation,
            "ObservationSpec",
            grid=grid,
            player_locations=player_locations,
            ghost_locations=ghost_locations,
            power_up_locations=power_up_locations,
            frightened_state_time=frightened_state_time,
            pellet_locations=pellet_locations,
            action_mask=action_mask,
            score=score,
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec.

        5 actions: [0,1,2,3,4] -> [Up, Right, Down, Left, No-op].

        Returns:
            action_spec: a `specs.DiscreteArray` spec object.
        """
        return specs.DiscreteArray(5, name="action")

    def __repr__(self) -> str:

        return (
            f"PacMan(\n"
            f"\tnum_rows={self.x_size!r},\n"
            f"\tnum_cols={self.y_size!r},\n"
            f"\ttime_limit={self.time_limit!r}, \n"
            f"\tgenerator={self.generator!r}, \n"
            ")"
        )

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment by calling the instance generator for a new instance.

        Args:
            key: A PRNGKey to use for random number generation.

        Returns:
            state: `State` object corresponding to the new state of the environment after a reset.
            timestep: `TimeStep` object corresponding the first timestep returned by the environment
                after a reset.
        """

        state = self.generator(key)

        # Generate observation
        obs = self._observation_from_state(state)

        # Return a restart timestep of step type is FIRST.
        timestep = restart(observation=obs)

        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of the environment's dynamics.

        If an action is invalid, the agent does not move, i.e. the episode does not
        automatically terminate.

        Args:
            state: State object containing the dynamics of the environment.
            action: (int32) specifying which action to take: [0,1,2,3,4] correspond to
                [Up, Right, Down, Left, No-op]. If an invalid action is taken, i.e. there is a wall
                blocking the action, then no action (no-op) is taken.

        Returns:
            state: the new state of the environment.
            the next timestep to be observed.
        """

        # Collect updated state based on environment dynamics
        updated_state, collision_rewards = self._update_state(state, action)

        # Create next_state from updated state
        next_state = updated_state.replace(step_count=state.step_count + 1)  # type: ignore

        # Check if episode terminates
        num_pellets = next_state.pellets
        dead = next_state.dead
        time_limit_exceeded = next_state.step_count >= self.time_limit
        all_pellets_found = num_pellets == 0
        dead = next_state.dead == 1
        done = time_limit_exceeded | dead | all_pellets_found

        reward = jnp.asarray(collision_rewards)
        # Generate observation from the state
        observation = self._observation_from_state(next_state)

        # Return either a MID or a LAST timestep depending on done.
        timestep = jax.lax.cond(
            done,
            termination,
            transition,
            reward,
            observation,
        )

        return next_state, timestep

    def _update_state(self, state: State, action: chex.Array) -> Tuple[State, int]:
        """Updates the state of the environment.

        Args:
            state: 'State` object corresponding to the new state of the environment.
            action: An integer representing the player action.

        Returns:
            state: 'State` object corresponding to the new state of the environment.
            collision_rewards: Rewards from objects the player has collided with
        """

        key = state.key
        key, _ = jax.random.split(key)

        # Move player
        next_player_pos = player_step(
            state=state, action=action, x_size=self.x_size, y_size=self.y_size, steps=1
        )
        next_player_pos = self.check_wall_collisions(state, next_player_pos)
        state = state.replace(last_direction=jnp.array(action, jnp.int32))  # type: ignore

        # Move ghosts
        old_ghost_locations = state.ghost_locations
        ghost_paths, ghost_actions, key = ghost_move(state, self.x_size, self.y_size)

        # Check for collisions with ghosts
        state, done, ghost_col_rewards = check_ghost_collisions(
            ghost_paths, next_player_pos, state
        )

        state = state.replace(player_locations=next_player_pos)  # type: ignore
        state = state.replace(dead=done)

        power_up_locations, eat, power_up_rewards = self.check_power_up(state)

        # Check for collected pellets
        collision_rewards, pellets, num_pellets = self.check_rewards(state)

        # Update old ghost locations
        state = state.replace(ghost_init_steps=state.ghost_init_steps - 1)
        state = state.replace(old_ghost_locations=old_ghost_locations)

        state = state.replace(pellet_locations=pellets)
        state = state.replace(pellets=num_pellets)
        state = state.replace(key=key)

        def powerup_collected() -> chex.Array:
            """If a power-up was collected set scatter time to 30"""
            return jnp.array(30, jnp.int32)

        def tick_frightened_time() -> chex.Array:
            """If a power-up was not collected reduce scatter time"""
            return jnp.array(state.frightened_state_time - 1, jnp.int32)

        # Check if frightened state is active and decrement timer
        state.frightened_state_time = jax.lax.cond(
            eat > 0, powerup_collected, tick_frightened_time
        )

        # Update power-up locations
        state.power_up_locations = power_up_locations
        state.ghost_actions = ghost_actions

        # Decrease ghost starting delay
        state.ghost_starts = state.ghost_starts - 1
        reward = collision_rewards + power_up_rewards + ghost_col_rewards
        state.score = jnp.array(state.score + reward, jnp.int32)
        return state, reward

    def check_rewards(self, state: State) -> Tuple[int, chex.Array, int]:
        """
        Process the state of the game to compute rewards, updated pellet spaces, and remaining
        number of pellets.

        Args:
            state: 'State` object corresponding to the current state of the environment

        Returns:
            rewards: an integer representing the reward earned by the player in the current state
            pellet_spaces: a 2D jax array showing the location of all remaining cookies
            num_cookies: an integer counting the remaining cookies on the map.
        """

        # Get the locations of the pellets and the player
        pellet_spaces = jnp.array(state.pellet_locations)
        player_space = state.player_locations
        ps = jnp.array([player_space.y, player_space.x])

        # Get the number of pellets on the map
        num_pellets = state.pellets

        # Check if player has eaten a pellet in this step
        ate_pellet = jnp.any(jnp.all(ps == pellet_spaces, axis=-1))

        # Reduce number of pellets on map if eaten, add reward and remove eaten pellet
        num_pellets -= ate_pellet.astype(int)
        rewards = ate_pellet * 10.0
        mask = jnp.logical_not(jnp.all(ps == pellet_spaces, axis=-1))
        pellet_spaces = pellet_spaces * mask[..., None]

        return rewards, pellet_spaces, num_pellets

    def player_step(self, state: State, action: int, steps: int = 1) -> Position:
        """
        Compute the new position of the player based on the given state and action.

        Args:
            state: 'state` object corresponding to the new state of the environment.
            action: an integer between 0 and 4 representing the player's chosen action.
            steps: how many steps ahead of current position to search.

        Returns:
            new_pos: a `Position` object representing the new position of the player.
        """

        position = state.player_locations

        move_left = lambda position: (position.y, position.x - steps)
        move_up = lambda position: (position.y - steps, position.x)
        move_right = lambda position: (position.y, position.x + steps)
        move_down = lambda position: (position.y + steps, position.x)
        no_op = lambda position: (position.y, position.x)

        new_pos_row, new_pos_col = jax.lax.switch(
            action, [move_left, move_up, move_right, move_down, no_op], position
        )

        new_pos = Position(x=new_pos_col % self.x_size, y=new_pos_row % self.y_size)
        return new_pos

    def check_power_up(
        self, state: State
    ) -> Tuple[chex.Array, chex.Numeric, chex.Numeric]:
        """
        Check if the player is on a power-up location and update the power-up
        locations array accordingly.

        Args:
            state: 'state` object corresponding to the new state of the environment

        Returns:
            power_up_locations: locations of the remaining power-ups
            eat: a bool indicating if the player can eat the ghosts
            reward: an int of the reward gained from collecting power-ups
        """

        power_up_locations = jnp.array(state.power_up_locations)

        player_space = state.player_locations
        player_loc_x = player_space.x
        player_loc_y = player_space.y
        player_loc = jnp.array([player_loc_y, player_loc_x])

        # Check if player and power_up position are shared
        on_powerup = (player_loc == power_up_locations).all(axis=-1).any()
        eat = on_powerup.astype(int)
        mask = (player_loc == power_up_locations).all(axis=-1)
        invert_mask = ~mask
        invert_mask = invert_mask.reshape(4, 1)

        # Mask out collected power-ups
        power_up_locations = power_up_locations * invert_mask

        # Assign reward for power-up
        reward = eat * 50.0

        return power_up_locations, eat, reward

    def check_wall_collisions(self, state: State, new_player_pos: Position) -> Any:

        """
        Check if the new player position collides with a wall.

        Args:
            state: 'State` object corresponding to the new state of the environment.
            new_player_pos: the position of the player after the last action.

        Returns:
            collision: a boolean indicating if the player has moved into a wall.
        """

        grid = state.grid
        location_value = grid[new_player_pos.x, new_player_pos.y]

        collision = jax.lax.cond(
            location_value == 1,
            lambda x: new_player_pos,
            lambda x: state.player_locations,
            0,
        )
        return collision

    def _compute_action_mask(self, state: State) -> chex.Array:
        """Compute the action mask.
        An action is considered invalid if it leads to a WALL or goes outside of the maze.
        """

        grid = state.grid
        player_pos = state.player_locations

        def is_move_valid(agent_position: Position, move: chex.Array) -> chex.Array:
            y, x = jnp.array([agent_position.y, agent_position.x]) + move
            return grid[x][y]

        # vmap over the moves.
        action_mask = jax.vmap(is_move_valid, in_axes=(None, 0))(
            player_pos, MOVES
        ) * jnp.array([True, True, True, True, False])

        return action_mask

    def _observation_from_state(self, state: State) -> Observation:
        """Create an observation from the state of the environment."""
        action_mask = self._compute_action_mask(state).astype(bool)
        return Observation(
            grid=state.grid,
            player_locations=state.player_locations,
            ghost_locations=state.ghost_locations,
            power_up_locations=state.power_up_locations,
            frightened_state_time=state.frightened_state_time,
            pellet_locations=state.pellet_locations,
            action_mask=action_mask,
            score=state.score,
        )

    def render(self, state: State) -> Any:
        """Render the given state of the environment.

        Args:
            state: `state` object containing the current environment state.
        """
        return self._viewer.render(state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Creates an animated gif of the `Maze` environment based on the sequence of states.

        Args:
            states: sequence of environment states corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            animation.FuncAnimation: the animation object that was created.
        """
        return self._viewer.animate(states, interval, save_path)

    def close(self) -> None:
        """Perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        self._viewer.close()
