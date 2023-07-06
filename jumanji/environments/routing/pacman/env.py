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
from jax import nn

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.pacman.constants import DEFAULT_MAZE
from jumanji.environments.routing.pacman.generator import AsciiGenerator, Generator
from jumanji.environments.routing.pacman.types import Observation, Position, State
from jumanji.environments.routing.pacman.utils import create_grid_image
from jumanji.environments.routing.pacman.viewer import PacManViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class PacMan(Environment[State]):
    """A JAX implementation of the 'Pac-Man' game where a single agent must navigate a
    maze to collect pellets and avoid 4 heuristic agents. The game takes place on a 31x28
    grid where the player can move in 4 directions (left, right, up, down) and collect
    pellets to gain points. The goal is to collect all of the pellets on the board without
    colliding with one of the heuristic agents.

    - observation: `Observation`
        - player_locations: current 2D position of agent.
        - grid: jax array (int)) of the ingame maze with walls.
        - ghost_locations: jax array (int) of ghost positions.
        - power_up_locations: jax array (int) of power-pellet locations
        - pellet_locations: jax array (int) of pellets.
        - action_mask: jax array (bool) defining current actions.

    - action: jax array (int) of shape () specifiying which action to take [0,1,2,3,4]
        corresponding to [up, right, down, left, no-op. If there is an invalid action
        taken, i.e. there is a wall blocking the action, then no action (no-op) is taken.

    - reward: jax array (float32) of shape (): 10 per pellet collected, 20 for a power pellet.

    - episode termination (if any):
        - agent has collected all pellets.
        - agent killed by ghost.

    - state: State:
        - key: jax array (uint32) of shape(2,).
        - grid: jax array (int)) of shape (300,2) of the ingame maze with walls.
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



    ```python
    from jumanji.environments import pac_man
    env = PacMan()
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
        viewer: Optional[Viewer[State]] = None,
        time_limit: Optional[int] = None,
    ) -> None:
        """Instantiates a `Pac-Man` environment.

        Args:
            generator: `Generator` whose `__call__` instantiates an environment instance.
                Implemented options are [`AsciiGenerator`].
            time_limit: the time_limit of an episode, i.e. the maximum number of environment steps
                before the episode terminates. By default, set to 2000.
            viewer: `Viewer` used for rendering. Defaults to `PacManViewer`.
        """

        self.generator = generator or AsciiGenerator(DEFAULT_MAZE)
        self.x_size = self.generator.x_size
        self.y_size = self.generator.y_size
        self.cookie_spaces = self.generator.cookie_spaces
        self._viewer = viewer or PacManViewer("Pacman", render_mode="human")
        self.time_limit = 2000

        self.generate_obs = create_grid_image

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specifications of the observation of the `PacMan` environment.

        Returns:
            Spec containing all the specifications for all the `Observation` fields:
            - player_locations: current 2D position of agent.
            - grid: jax array (int)) of the ingame maze with walls.
            - ghost_locations: jax array (int) of ghost positions.
            - power_up_locations: jax array (int) of power-pellet locations
            - pellet_locations: jax array (int) of pellets.
            - action_mask: jax array (bool) defining current actions.
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
        fruit_locations = specs.Array(
            shape=(self.cookie_spaces.shape[0], self.cookie_spaces.shape[1]),
            dtype=jnp.int32,
            name="fruit_locations",
        )
        action_mask = specs.BoundedArray(
            shape=(5,),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask",
        )

        frightened_state_time = specs.Array((), jnp.int32, "frightened_state_time")

        return specs.Spec(
            Observation,
            "ObservationSpec",
            grid=grid,
            player_locations=player_locations,
            ghost_locations=ghost_locations,
            power_up_locations=power_up_locations,
            frightened_state_time=frightened_state_time,
            fruit_locations=fruit_locations,
            action_mask=action_mask,
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec.

        5 actions: [0,1,2,3,4] -> [Up, Right, Down, Left, No-op].

        Returns:
            action_spec: a `specs.DiscreteArray` spec object.
        """
        return specs.DiscreteArray(
            5,
            name="action",
        )

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
        """
        Reset the game to its initial state.

        Args:
            key: A PRNGKey to use for random number generation.

        Returns:
            state: the new state of the environment.
            timestep: the first timestep returned by the environment.
        """

        state = self.generator(key)

        # Generate the observation and initial timestep
        action_mask = jnp.array([True, True, True, True, False])
        obs = Observation(
            grid=state.grid,
            player_locations=state.player_locations,
            ghost_locations=state.ghost_locations,
            power_up_locations=state.power_up_locations,
            frightened_state_time=state.frightened_state_time,
            fruit_locations=state.fruit_locations,
            action_mask=action_mask,
        )

        timestep = restart(observation=obs)

        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        """Updates the environment state after the agent takes an action.

        Args:
            state: the current state of the environment.
            action: the action taken by the agent.

        Returns:
            state: the new state of the environment.
            timestep: the next timestep.
        """

        updated_state, collision_rewards = self._update_state(state, action)
        next_state = State(
            key=updated_state.key,
            grid=updated_state.grid,
            pellets=updated_state.pellets,
            frightened_state=updated_state.frightened_state,
            frightened_state_time=updated_state.frightened_state_time,
            fruit_locations=updated_state.fruit_locations,
            power_up_locations=updated_state.power_up_locations,
            player_locations=updated_state.player_locations,
            ghost_locations=updated_state.ghost_locations,
            old_ghost_locations=updated_state.old_ghost_locations,
            initial_player_locations=updated_state.initial_player_locations,
            initial_ghost_positions=updated_state.initial_ghost_positions,
            last_direction=updated_state.last_direction,
            dead=updated_state.dead,
            ghost_init_steps=updated_state.ghost_init_steps,
            ghost_init_targets=updated_state.ghost_init_targets,
            ghost_actions=updated_state.ghost_actions,
            visited_index=updated_state.visited_index,
            ghost_starts=updated_state.ghost_starts,
            scatter_targets=updated_state.scatter_targets,
            step_count=state.step_count + 1,
        )
        num_pellets = state.pellets
        dead = state.dead

        # Check if episode terminates
        time_limit_exceeded = state.step_count >= self.time_limit
        all_pellets_found = num_pellets == 0
        dead = state.dead == 1
        done = time_limit_exceeded | dead | all_pellets_found

        reward = jnp.asarray(collision_rewards)

        action_mask = jnp.array([True, True, True, True, False])

        # Generate observation from the state
        observation = Observation(
            grid=state.grid,
            player_locations=state.player_locations,
            ghost_locations=state.ghost_locations,
            power_up_locations=state.power_up_locations,
            frightened_state_time=state.frightened_state_time,
            fruit_locations=state.fruit_locations,
            action_mask=action_mask,
        )

        # Return either a MID or a LAST timestep depending on done.
        timestep = jax.lax.cond(
            done,
            lambda: termination(
                reward=reward,
                observation=observation,
            ),
            lambda: transition(
                reward=reward,
                observation=observation,
            ),
        )

        return next_state, timestep

    def _update_state(self, state: State, action: chex.Array) -> Tuple[State, int]:
        """
        Updates the state of the environment.

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
        next_player_pos = self.player_step(state=state, action=action, steps=1)
        next_player_pos = self.check_wall_collisions(state, next_player_pos)
        state.player_locations = next_player_pos

        ## Move ghosts
        def call_ghost_step(state: State) -> Tuple[chex.Array, int, chex.PRNGKey]:
            ghost_paths, ghost_actions, key = self.ghost_move(state)
            return ghost_paths, ghost_actions, key

        old_ghost_locations = state.ghost_locations
        ghost_paths, ghost_actions, key = call_ghost_step(state)

        # Check for collisions with ghosts
        state, done, ghost_col_rewards = self.check_ghost_collisions(
            ghost_paths, next_player_pos, state
        )
        state.dead = done
        power_up_locations, eat, power_up_rewards = self.check_power_up(state)

        # Check for collected pellets
        collision_rewards, cookie_list, num_cookies = self.check_rewards(state)

        # Update old ghost locations
        state.ghost_init_steps = state.ghost_init_steps - 1
        state.old_ghost_locations = old_ghost_locations

        state.fruit_locations = cookie_list
        state.pellets = num_cookies
        state.key = key
        # state.frightened_state = eat

        def f_time() -> Any:
            """If in scatter mode then decrement timer or add to time if eaten a pellet"""
            frightened_state_time = jnp.array(30, jnp.int32)
            return frightened_state_time

        def ff_time() -> Any:
            """If not in scatter mode then leave unchanged"""
            frightened_state_time = jnp.array(
                state.frightened_state_time - 1, jnp.int32
            )
            return frightened_state_time

        # Check if frightened state is active and decrement timer
        state.frightened_state_time = jax.lax.cond(eat > 0, f_time, ff_time)

        # Update power up locations
        state.power_up_locations = power_up_locations
        state.ghost_actions = ghost_actions

        state.ghost_starts = state.ghost_starts - 1
        reward = collision_rewards + power_up_rewards + ghost_col_rewards
        return state, reward

    def ghost_move(self, state: State) -> Tuple[chex.Array, chex.Array, chex.PRNGKey]:
        """
        Generate a random move for each ghost and update their positions accordingly.

        Args:
            state: 'State` object corresponding to the new state of the environment.

        Returns:
            ghost_path: The new positions of each ghost after moving.
            key: New PRNG key.
        """

        key = state.key
        key, subkey0, subkey1, subkey2, subkey3 = jax.random.split(key, 5)
        ghost_keys = jnp.array([subkey0, subkey1, subkey2, subkey3])
        ghost_pos = state.ghost_locations
        player_pos = state.player_locations
        start_time = state.ghost_starts
        ghost_paths = []
        scatter_targets = state.scatter_targets
        actions = jnp.array(state.ghost_actions)

        def move(
            ghost_pos: chex.Array,
            subkey: chex.PRNGKey,
            ghost_action: int,
            ghost_num: int,
            player_pos: chex.Array,
            ghost_init_target: chex.Array,
            old_ghost_locations: chex.Array,
            ghost_start: chex.Array,
            scatter_target: chex.Array,
        ) -> Tuple[chex.Array, int]:
            """Move a single ghost"""
            act = jnp.array([0, 1, 2, 3])

            subkey, subsubkey = jax.random.split(subkey)
            _, valids, valid_actions = self.check_ghost_wall_collisions(
                state,
                ghost_pos,
                ghost_num,
                player_pos,
                ghost_init_target,
                old_ghost_locations,
                scatter_target,
            )
            vert_col = jnp.array([1, 0, 1, 0])
            hor_col = jnp.array([0, 1, 0, 1])

            # If valids is  [1,0,1,0] or [0,1,0,1] then use old action
            # this is the case where the ghosts are in a tunnel as ghosts
            # are not allowed to backtrack.
            condition = jnp.logical_or(
                jnp.array_equal(valids, vert_col), jnp.array_equal(valids, hor_col)
            )
            condition = jnp.all(condition)

            def is_tunnel(inputs: Tuple[chex.Array, chex.Array, int]) -> int:
                """Repeat old action if in tunnel"""
                _, _, ghost_action = inputs
                return ghost_action

            def no_tunnel(
                inputs: Tuple[chex.Array, chex.Array, int, chex.PRNGKey, int]
            ) -> Any:
                """Chose new action when at intersection"""
                valids, act, _, key0, _ = inputs
                key0, subkey0 = jax.random.split(key0)
                logits = valids
                logits2 = jnp.where(logits == 0, -1e9, logits)
                # Softmax distribution
                softmax_dist = nn.softmax(logits2)
                a2 = act * logits
                new_action = jax.random.choice(subkey0, a2, p=softmax_dist).astype(int)
                return new_action

            inputs_tunnel = (valid_actions, act, ghost_action)
            inputs_no_tunnel = (valid_actions, act, ghost_action, subsubkey, ghost_num)

            result_true = is_tunnel(inputs_tunnel)
            result_false = no_tunnel(inputs_no_tunnel)

            def start_over() -> Any:
                """If not in waiting mode then pick new action"""
                chosen_action = jnp.where(
                    condition[..., None], result_true, result_false
                )
                chosen_action = jnp.squeeze(chosen_action)
                return chosen_action

            def no_start() -> int:
                """If in waiting mode then use no-op"""
                chosen_action = 4
                return chosen_action

            position = ghost_pos
            chosen_action = jax.lax.cond(ghost_start < 0, start_over, no_start)

            # Use chosen action
            move_left = lambda position: (position[1], position[0] - 1)
            move_up = lambda position: (position[1] - 1, position[0])
            move_right = lambda position: (position[1], position[0] + 1)
            move_down = lambda position: (position[1] + 1, position[0])
            no_op = lambda position: (position[1], position[0])

            new_pos_row, new_pos_col = jax.lax.switch(
                chosen_action,
                [move_left, move_up, move_right, move_down, no_op],
                position,
            )

            # If using the teleporter shift o the other side of the map
            path = jnp.array([new_pos_col % self.x_size, new_pos_row % self.y_size])
            path = jax.lax.cond(ghost_start <= 0, lambda: path, lambda: position)

            return path, chosen_action

        ghost_nums = jnp.array([0, 1, 2, 3])
        ghost_init_targets = state.ghost_init_targets
        ghost_paths, ghost_acts = jax.vmap(
            move, in_axes=(0, 0, 0, 0, None, 0, 0, 0, 0)
        )(
            jnp.array(ghost_pos),
            ghost_keys,
            actions,
            ghost_nums,
            player_pos,
            jnp.array(ghost_init_targets),
            jnp.array(state.old_ghost_locations),
            start_time,
            scatter_targets,
        )

        return ghost_paths, ghost_acts, key

    def check_rewards(self, state: State) -> Tuple[int, chex.Array, int]:
        """Process the state of the game to compute rewards, updated pellet spaces, and remaining number of
        pellets.

        Args:
            state: 'state` object corresponding to the current state of the environment

        Returns:
            rewards: a integer representing the reward earned by the player in the current state
            cookie_spaces: a 2D jax array showing the location of all remaining cookies
            num_cookies: an integer counting the remaining cookies on the map.

        """
        cookie_spaces = jnp.array(state.fruit_locations)
        player_space = state.player_locations
        ps = jnp.array([player_space.y, player_space.x])

        num_cookies = state.pellets
        valid = jnp.any(jnp.all(ps == cookie_spaces, axis=-1))

        num_cookies -= valid * 1
        rewards = valid * 10.0
        mask = jnp.logical_not(jnp.all(ps == cookie_spaces, axis=-1))
        cookie_spaces = cookie_spaces * mask[..., None]

        return rewards, cookie_spaces, num_cookies

    def player_step(self, state: State, action: int, steps: int = 1) -> Position:
        """
        Compute the new position of the player based on the given state and action.

        Args:
            state: 'state` object corresponding to the new state of the environment
            action: an integer between 0 and 4 representing the player's chosen action

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
        ps_x = player_space.x
        ps_y = player_space.y
        ps = jnp.array([ps_y, ps_x])
        valid = (ps == power_up_locations).all(axis=-1).any()
        eat = 1 * valid
        mask = (ps == power_up_locations).all(axis=-1)
        invert_mask = mask != True  # type: ignore # noqa: E712
        invert_mask = invert_mask.reshape(4, 1)
        power_up_locations = power_up_locations * invert_mask
        reward = eat * 50.0

        return power_up_locations, eat, reward

    def check_wall_collisions(self, state: State, new_player_pos: Position) -> Any:

        """
        Check if the new player position collides with a wall.

        Args:
            state: 'state` object corresponding to the new state of the environment.
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

    def check_ghost_wall_collisions(
        self,
        state: State,
        new_player_pos: chex.Array,
        ghost_num: int,
        pacman_pos: Position,
        init_target: chex.Array,
        old_ghost_locations: chex.Array,
        scatter_target: chex.Array,
    ) -> Tuple[bool, chex.Array, chex.Array]:

        grid = jnp.array(state.grid)
        pacman_pos = Position(x=pacman_pos.y, y=pacman_pos.x)
        pac_dir = state.last_direction
        is_scared = state.frightened_state_time  # [0]
        ghost_init_steps = state.ghost_init_steps

        # Get all possible positions
        old_ghost_location = jnp.array([old_ghost_locations[1], old_ghost_locations[0]])
        a0 = new_player_pos[1], new_player_pos[0] - 1
        a1 = new_player_pos[1] - 1, new_player_pos[0]
        a2 = new_player_pos[1], new_player_pos[0] + 1
        a3 = new_player_pos[1] + 1, new_player_pos[0]
        ghost_p = jnp.array([a0, a1, a2, a3])

        # Block old paths so ghosts don't backtrack
        ghost_mask = jnp.any(ghost_p != old_ghost_location, axis=1)

        def get_directions(
            pacman_position: Position, ghost_position: chex.Array
        ) -> chex.Array:

            distance = jnp.array(
                [
                    ghost_position[0] - pacman_position.y,
                    ghost_position[1] - pacman_position.x,
                ]
            )

            return distance

        def get_distances(distance_list: chex.Array) -> chex.Array:

            d0 = jnp.linalg.norm(distance_list[0])
            d1 = jnp.linalg.norm(distance_list[1])
            d2 = jnp.linalg.norm(distance_list[2])
            d3 = jnp.linalg.norm(distance_list[3])
            ghost_dist = jnp.array([d0, d1, d2, d3])
            return ghost_dist

        # For ghost 0: Move to closest tile to pacman
        def chase_ghost(pacman_pos: Position) -> Tuple[chex.Array, chex.Array]:

            distance_list = jax.vmap(get_directions, in_axes=(None, 0))(
                pacman_pos, ghost_p
            )
            ghost_dist = get_distances(distance_list)
            return distance_list, ghost_dist

        # For ghost 1: move 4 steps ahead of pacman
        def block_ghost(
            pacman_pos: Position, steps: int = 4
        ) -> Tuple[chex.Array, chex.Array]:
            pac_pos = self.player_step(state, pac_dir, steps=steps)
            distance_list = jax.vmap(get_directions, in_axes=(None, 0))(
                pac_pos, ghost_p
            )
            ghost_dist = get_distances(distance_list)

            return distance_list, ghost_dist

        # For ghost 2: move to averge between 0 and 1
        def smart_ghost(pacman_pos: Position) -> Tuple[chex.Array, chex.Array]:
            ghost_0_target, _ = chase_ghost(pacman_pos)
            ghost_1_target, _ = block_ghost(pacman_pos=pacman_pos, steps=4)
            distance_list = ghost_0_target + ghost_1_target
            ghost_dist = get_distances(distance_list)
            return distance_list, ghost_dist

        def random_ghost(pacman_pos: Position) -> Tuple[chex.Array, chex.Array]:

            distance_pacman = jnp.array(
                [new_player_pos[0] - pacman_pos.x, new_player_pos[1] - pacman_pos.y]
            )
            distance_pacman = jnp.linalg.norm(distance_pacman)

            distance_list = jax.vmap(get_directions, in_axes=(None, 0))(
                pacman_pos, ghost_p
            )

            _, ghost_dist = jax.lax.cond(
                distance_pacman > 8, chase_ghost, scared_behaviors, pacman_pos
            )

            return distance_list, ghost_dist

        def general_behaviors(pacman_pos: Position) -> Tuple[chex.Array, chex.Array]:
            blinky = lambda pacman_pos: chase_ghost(pacman_pos)
            inky = lambda pacman_pos: block_ghost(pacman_pos)
            pinky = lambda pacman_pos: smart_ghost(pacman_pos)
            clyde = lambda pacman_pos: random_ghost(pacman_pos)
            _, ghost_dist = jax.lax.switch(
                ghost_num, [blinky, inky, pinky, clyde], pacman_pos
            )
            return _, ghost_dist

        def scared_behaviors(pacman_pos: Position) -> Tuple[chex.Array, chex.Array]:
            scatter_pos = Position(x=scatter_target[1], y=scatter_target[0])
            _, ghost_dist = chase_ghost(scatter_pos)
            return _, ghost_dist

        def behaviors() -> Tuple[chex.Array, chex.Array]:
            _, ghost_dist = jax.lax.cond(
                is_scared > 0, scared_behaviors, general_behaviors, pacman_pos
            )
            return _, ghost_dist

        def init_behaviors() -> Tuple[chex.Array, chex.Array]:
            target = Position(y=init_target[1], x=init_target[0])
            _, ghost_dist = chase_ghost(pacman_pos=target)
            return _, ghost_dist

        _, ghost_dist = jax.lax.cond(
            ghost_init_steps[ghost_num] > 0, init_behaviors, behaviors
        )

        def get_valid_positions(pos: chex.Array) -> Any:

            value = grid[pos[0]][pos[1]]
            return value

        valids = jax.vmap(get_valid_positions, in_axes=(0,))(ghost_p)
        valid_locs = valids

        # Set all invalid areas to an infinite cost
        # remove back tracking
        valid_no_back = valids * ghost_mask
        # Get distances of valid locations
        valid_no_back_d = valid_no_back * ghost_dist
        invert_mask = valid_no_back != True  # type: ignore # noqa: E712
        invert_mask = invert_mask * jnp.inf
        # Set distance of all invalid areas to infinity
        valid_no_back_d = valid_no_back_d + invert_mask
        masked_dist = valid_no_back_d
        minimum_distance = jnp.min(masked_dist)
        # set all other values aside from the minimum to  0 and the minimum to 1
        masked_dist = jnp.where(masked_dist == minimum_distance, 1, 0)

        location_value = grid[new_player_pos[1]][new_player_pos[0]]
        # Check if colliding with walls
        false_cond = lambda x: False
        true_cond = lambda x: True
        collision = jax.lax.cond(location_value == 1, false_cond, true_cond, 0)
        return collision, valid_locs, masked_dist

    def check_ghost_collisions(
        self, ghost_pos: chex.Array, new_player_pos: chex.Array, state: State
    ) -> Tuple[State, bool, chex.Numeric]:
        """
        Check if the player collides with a ghost.

        Args:
            ghost_pos: A list of tuples representing the current positions of the ghosts.
            new_player_pos: The current position of the player.
            state: 'State` object corresponding to the new state of the environment.

        Returns:
            is_col: Did the player collide with the ghost.
            state: 'State` object corresponding to the new state of the environment.
        """

        og_pos = state.initial_ghost_positions

        def check_collisions(
            ghost_pos: chex.Array,
            new_player_pos: chex.Array,
            og_pos: chex.Array,
            state: State,
        ) -> Tuple[chex.Array, chex.Numeric, chex.Numeric, chex.Numeric]:

            eat = lambda: True
            no_eat = lambda: False
            frightened_time = state.frightened_state_time

            is_eat = jax.lax.cond(frightened_time > 0, eat, no_eat)

            ghost_p = Position(y=ghost_pos[0], x=ghost_pos[1])
            # Check for collision if moving to the same space
            cond_x1 = ghost_p.x == new_player_pos.x
            cond_y1 = ghost_p.y == new_player_pos.y
            cond1 = cond_x1 * cond_y1

            # Check for collision if ghost moves to space when player and ghost are adjacent
            cond_x2 = ghost_p.x == state.player_locations.x
            cond_y2 = ghost_p.y == state.player_locations.y
            cond2 = cond_x2 * cond_y2
            cond = cond1 * cond2

            ghost_reset = is_eat * cond
            ghost_init_steps = ghost_reset * 6

            def no_col_fn() -> Tuple[chex.Array, chex.Numeric, chex.Numeric]:
                return ghost_pos, False, 0.0

            def col_fn() -> Tuple[chex.Array, chex.Numeric, chex.Numeric]:
                reset_true = lambda: (jnp.array(og_pos), False, 200.0)
                reset_false = lambda: (ghost_pos, True, 0.0)
                path, done, col_reward = jax.lax.cond(
                    ghost_reset, reset_true, reset_false
                )
                return path, done, col_reward

            # First check for collision
            path, done, col_reward = jax.lax.cond(cond, col_fn, no_col_fn)
            return path, ghost_init_steps, done, col_reward

        ghost_positions, ghost_init, dones, col_rewards = jax.vmap(
            check_collisions, in_axes=(0, None, 0, None)
        )(ghost_pos, new_player_pos, jnp.array(og_pos), state)
        done = jnp.any(dones)
        col_rewards = col_rewards.sum()
        state.ghost_locations = ghost_positions
        state.ghost_init_steps = jnp.array(ghost_init, int)

        return state, done, col_rewards

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
