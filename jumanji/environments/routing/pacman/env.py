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
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer
from jumanji.environments.pacman.types import Observation, Position, State
from jumanji.environments.pacman.utils import convert_maze_to_numpy, create_grid_image
from jumanji.environments.pacman.viewer import PacManViewer


class PacMan(Environment[State]):
    """A JAX implementation of the 'Pac-Man' game where a single agent must navigate a
    maze to collect pellets and avoid 4 heuristic agents

    ```python
    from jumanji.environments import pac_man
    env = PacMan()
    key = jax.random.key(0)
    state, timestep = jax.jit(env.reset)(key)
    env.render(state) #TODO
    action = env.action_spec().generate_value()
    state, timestep = jax.jit(env.step)(state, action)
    env.render(state) #TODO
    ```
    """

    def __init__(self, viewer: Optional[Viewer[State]] = None) -> None:
        """Instantiates a `Pac-Man` environment."""

        self.map_data = convert_maze_to_numpy()
        self.numpy_maze = jnp.array(self.map_data[0])
        self.cookie_spaces = jnp.array(self.map_data[1])
        self.powerup_spaces = jnp.array(self.map_data[2])
        self.reachable_spaces = self.map_data[3]

        self.ghost_spawns = jnp.array(self.map_data[4])
        self.player_coords = Position(y=self.map_data[5][0], x=self.map_data[5][1])
        self.init_targets = self.map_data[6]
        self.scatter_targets = jnp.array(self.map_data[7])
        self._viewer = viewer or PacManViewer("Pacman", render_mode="human")
        self.generate_obs = create_grid_image

    def observation_spec(self) -> specs.BoundedArray:
        """Returns the observation spec.
        Returns:
            Spec for the `Observation` whose fields are:
            - grid: BoundedArray (float) of shape (num_rows, num_cols, 3).
        """

        return specs.BoundedArray(
            shape=(31, 28, 3),
            dtype=float,
            name="observation",
            minimum=0,
            maximum=1,
        )
        return

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec. 5 actions: [0,1,2,3,4] -> [Up, Right, Down, Left, No-op].
        Returns:
            action_spec: a `specs.DiscreteArray` spec.
        """
        return specs.DiscreteArray(
            5,
            name="action",
        )
    
    def __repr__(self) -> str:
        return (
            f"Pacman"
        )

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """
        Reset the game to its initial state.

        Args:
            key: A PRNGKey to use for random number generation.

        Returns:
            A tuple containing the updated game state and the initial timestep information:
            (state, timestep).
        """

        # Initialize game state variables
        grid = self.numpy_maze
        pellets = 320
        frightened_state = 0
        frightened_state_time = 0
        fruit_locations = self.cookie_spaces
        power_up_locations = self.powerup_spaces
        player_locations = self.player_coords
        ghost_locations = self.ghost_spawns
        last_direction = 0
        ghost_init_steps = jnp.array([0, 0, 0, 0])
        ghost_init_targets = self.init_targets
        ghost_actions = jnp.array([1, 1, 1, 1])
        old_ghost_locations = ghost_locations

        # Create a new `State` object with the initial game state
        state = State(
            key=key,
            grid=grid,
            pellets=pellets,
            frightened_state=frightened_state,
            frightened_state_time=frightened_state_time,
            fruit_locations=fruit_locations,
            power_up_locations=power_up_locations,
            player_locations=player_locations,
            ghost_locations=ghost_locations,
            old_ghost_locations=old_ghost_locations,
            initial_player_locations=player_locations,
            initial_ghost_positions=ghost_locations,
            last_direction=last_direction,
            dead=False,
            ghost_init_steps=ghost_init_steps,
            ghost_init_targets=ghost_init_targets,
            ghost_actions=ghost_actions,
            visited_index=player_locations,
            ghost_starts=jnp.array([0, 10, 20, 30]),
            scatter_targets=self.scatter_targets,
        )

        # Generate the observation and initial timestep
        observation = self.generate_obs(state)
        timestep = restart(observation=observation)

        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        """
        Run one timestep of the game.

        Args:
            state: 'State` object corresponding to the new state of the environment.
            action: An integer representing the player action.

        Returns:
            state, timestep: next state of the environment and timestep to be observed.
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
        )
        num_pellets = state.pellets
        dead = state.dead
        done = (num_pellets == 0) | (dead == 1)
        reward = jnp.asarray(collision_rewards)
        observation = self.generate_obs(state)

        timestep = jax.lax.cond(done, termination, transition, reward, observation)
        jax.debug.print("timestep: {y}", y=timestep.last())

        return next_state, timestep

    def _update_state(self, state: State, action: chex.Array) -> Tuple[State, int]:
        """
        Update the state of the game based on the player action and ghost movements.

        Args:
            state: 'State` object corresponding to the new state of the environment.
            action: An integer representing the player action.

        Returns:
            state: 'State` object corresponding to the new state of the environment.
            collision_rewards: Rewards from objects the player has collided with
        """

        key = state.key
        key, subkey = jax.random.split(key)

        # Move player
        next_player_pos = self.player_step(state, action)
        next_player_pos = self.check_wall_collisions(state, next_player_pos)
        state.player_locations = next_player_pos

        ## Move ghosts
        def call_random_step(state: State) -> Tuple[chex.Array, int, chex.PRNGKey]:
            ghost_paths, ghost_actions, key = self.random_move(state)
            return ghost_paths, ghost_actions, key

        old_ghost_locations = state.ghost_locations
        ghost_paths, ghost_actions, key = call_random_step(state)

        # Check for collisions with ghosts
        state, done = self.check_ghost_collisions(ghost_paths, next_player_pos, state)
        state.dead = done
        power_up_locations, eat = self.check_power_up(state)
        # jax.debug.print("eat: {y}",y=eat)
        collision_rewards, cookie_list, num_cookies = self.check_rewards(state)
        #
        state.ghost_init_steps = state.ghost_init_steps - 1
        state.old_ghost_locations = old_ghost_locations

        state.fruit_locations = cookie_list
        state.pellets = num_cookies
        state.key = key
        state.frightened_state = eat

        def f_time() -> Any:
            state.frightened_state_time = 60 * eat + state.frightened_state_time - 1
            return state.frightened_state_time

        def ff_time() -> Any:
            state.frightened_state_time = 0
            return state.frightened_state_time

        state.frightened_state_time = jax.lax.cond(
            state.frightened_state_time >= 0, f_time, ff_time
        )

        state.power_up_locations = power_up_locations
        state.ghost_actions = ghost_actions

        state.ghost_starts = state.ghost_starts - 1
        return state, collision_rewards

    def random_move(self, state: State) -> Tuple[chex.Array, chex.Array, chex.PRNGKey]:
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
            # Check valid
            # If valids is  [1,0,1,0] or [0,1,0,1] then use old action
            condition = jnp.logical_or(
                jnp.array_equal(valids, vert_col), jnp.array_equal(valids, hor_col)
            )
            condition = jnp.all(condition)

            # True cond
            def is_tunnel(inputs: Tuple[chex.Array, chex.Array, int]) -> int:
                _, _, ghost_action = inputs
                return ghost_action

            # False cond
            def no_tunnel(
                inputs: Tuple[chex.Array, chex.Array, int, chex.PRNGKey, int]
            ) -> Any:
                valids, act, _, key0, _ = inputs
                key0, subkey0 = jax.random.split(key0)
                logits = valids
                logits2 = jnp.where(logits == 0, -1e9, logits)
                # Softmax distribution
                softmax_dist = nn.softmax(logits2)
                a2 = act * logits
                new_action = jax.random.choice(subkey0, a2, p=softmax_dist).astype(int)
                return new_action

            inputs_true = (valid_actions, act, ghost_action)
            inputs_false = (valid_actions, act, ghost_action, subsubkey, ghost_num)

            result_true = is_tunnel(inputs_true)
            result_false = no_tunnel(inputs_false)

            def start_over() -> Any:
                chosen_action = jnp.where(
                    condition[..., None], result_true, result_false
                )
                # chosen_action = no_tunnel(inputs_false)
                chosen_action = jnp.squeeze(chosen_action)
                return chosen_action

            def no_start() -> int:
                chosen_action = 4
                return chosen_action

            # chosen_action = is_tunnel(inputs_true)
            position = ghost_pos

            chosen_action = jax.lax.cond(ghost_start < 0, start_over, no_start)
            #jax.debug.print("chosen_action: {y}", y=chosen_action)
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

            path = jnp.array([new_pos_col % 31, new_pos_row % 28])

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
        """Process the state of the game to compute rewards, updated cookie spaces, and remaining number of
        cookies.

        Args:
            state: 'State` object corresponding to the new state of the environment

        Returns:
            rewards: A integer representing the reward earned by the player in the current state
            cookie_spaces: A 2D array showing the location of all remaining cookies
            num_cookies: An integer counting the remaining cookies on the map.

        """
        cookie_spaces = jnp.array(state.fruit_locations)
        player_space = state.player_locations
        ps = jnp.array([player_space.y, player_space.x])

        num_cookies = state.pellets
        valid = jnp.any(jnp.all(ps == cookie_spaces, axis=-1))

        num_cookies -= valid * 1
        rewards = valid * 10
        mask = jnp.logical_not(jnp.all(ps == cookie_spaces, axis=-1))
        cookie_spaces = cookie_spaces * mask[..., None]

        return rewards, cookie_spaces, num_cookies

    def player_step(self, state: State, action: int) -> Position:
        """
        Compute the new position of the player based on the given state and action.

        Args:
            state: 'State` object corresponding to the new state of the environment
            action: An integer between 0 and 4 representing the player's chosen action

        Returns:
            new_pos: A `Position` object representing the new position of the player.
        """

        position = state.player_locations

        move_left = lambda position: (position.y, position.x - 1)
        move_up = lambda position: (position.y - 1, position.x)
        move_right = lambda position: (position.y, position.x + 1)
        move_down = lambda position: (position.y + 1, position.x)
        no_op = lambda position: (position.y, position.x)

        new_pos_row, new_pos_col = jax.lax.switch(
            action, [move_left, move_up, move_right, move_down, no_op], position
        )

        new_pos = Position(x=new_pos_col % 31, y=new_pos_row % 28)
        return new_pos

    def player_step_look(self, state: State, action: int, steps: int) -> Position:
        """
        Compute the new position of the player based on the given state and action.

        Args:
            state: 'State` object corresponding to the new state of the environment
            action: An integer between 0 and 4 representing the player's chosen action

        Returns:
            new_pos: A `Position` object representing the new position of the player.
        """

        position = state.player_locations

        move_left = lambda position: (position.y, position.x - steps)
        move_up = lambda position: (position.y - steps, position.x)
        move_right = lambda position: (position.y, position.x + steps)
        move_down = lambda position: (position.y + steps, position.x)

        new_pos_row, new_pos_col = jax.lax.switch(
            action - 1, [move_left, move_up, move_right, move_down], position
        )

        new_pos = Position(x=new_pos_col, y=new_pos_row)
        return new_pos

    def check_power_up(self, state: State) -> Tuple[chex.Array, int]:
        """
        Check if the player is on a power-up location and update the power-up
        locations array accordingly.

        Args:
            state: 'State` object corresponding to the new state of the environment

        Returns:
            power_up_locations: Locations of the remaining power-ups
            eat: A bool indicating if the player can eat the ghosts
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
        return power_up_locations, eat

    def check_wall_collisions(self, state: State, new_player_pos: Position) -> Any:

        """
        Check if the new player position collides with a wall.

        Args:
            state: 'State` object corresponding to the new state of the environment.
            new_player_pos: The position of the player after the last action.

        Returns:
            collision: A boolean indicating if the player has moved into a wall.
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
        # new_plyaer_pos is just ghost_pos***

        grid = jnp.array(state.grid)
        pacman_pos = Position(x=pacman_pos.y, y=pacman_pos.x)
        pac_dir = state.last_direction
        is_scared = state.frightened_state_time
        ghost_init_steps = state.ghost_init_steps

        # Get all possible positions
        old_ghost_location = jnp.array([old_ghost_locations[1], old_ghost_locations[0]])
        # print(old_ghost_location)
        a0 = new_player_pos[1], new_player_pos[0] - 1
        a1 = new_player_pos[1] - 1, new_player_pos[0]
        a2 = new_player_pos[1], new_player_pos[0] + 1
        a3 = new_player_pos[1] + 1, new_player_pos[0]
        ghost_p = jnp.array([a0, a1, a2, a3])

        # Block old paths so ghosts don't backtrack
        ghost_mask = jnp.any(ghost_p != old_ghost_location, axis=1)
        #jax.debug.print("ghost_mask: {y}", y=ghost_mask)

        def get_directions(
            pacman_position: Position, ghost_position: chex.Array
        ) -> chex.Array:

            a0 = ghost_position[0]
            a1 = ghost_position[1]
            a2 = ghost_position[2]
            a3 = ghost_position[3]

            l0 = jnp.array([a0[0] - pacman_position.y, a0[1] - pacman_position.x])
            l1 = jnp.array([a1[0] - pacman_position.y, a1[1] - pacman_position.x])
            l2 = jnp.array([a2[0] - pacman_position.y, a2[1] - pacman_position.x])
            l3 = jnp.array([a3[0] - pacman_position.y, a3[1] - pacman_position.x])

            distance_list = jnp.array([l0, l1, l2, l3])

            return distance_list

        def get_distances(distance_list: chex.Array) -> chex.Array:

            d0 = jnp.linalg.norm(distance_list[0])
            d1 = jnp.linalg.norm(distance_list[1])
            d2 = jnp.linalg.norm(distance_list[2])
            d3 = jnp.linalg.norm(distance_list[3])
            ghost_dist = jnp.array([d0, d1, d2, d3])
            return ghost_dist

        # For ghost 0: Move to closest tile to pacman
        def chase_ghost(pacman_pos: Position) -> Tuple[chex.Array, chex.Array]:
            distance_list = get_directions(pacman_pos, ghost_p)
            ghost_dist = get_distances(distance_list)
            return distance_list, ghost_dist

        # For ghost 1: move 4 steps ahead of pacman
        def block_ghost(
            pacman_pos: Position, steps: int = 4
        ) -> Tuple[chex.Array, chex.Array]:
            pac_pos = self.player_step_look(state, pac_dir, steps)
            distance_list = get_directions(pac_pos, ghost_p)
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

            distance_list = get_directions(pacman_pos, ghost_p)

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
        # jax.debug.print("ghost_p: {y}",y=ghost_p)
        # jax.debug.print("valids: {y}",y=valids)

        # invert_mask = mask != True  # type: ignore # noqa: E712
        # invert_mask = invert_mask.reshape(4, 1)
        # Set all invalid areas to an infinite cost
        # remove back tracking
        valid_no_back = valids * ghost_mask
        # jax.debug.print("valid_no_back: {y}",y=valid_no_back)
        # Get distances of valid locations
        valid_no_back_d = valid_no_back * ghost_dist
        # jax.debug.print("valid_no_back_d: {y}",y=valid_no_back_d)
        invert_mask = valid_no_back != True  # type: ignore # noqa: E712
        # jax.debug.print("invert_mask: {y}",y=invert_mask)
        # invert_mask = invert_mask.reshape(4, 1) * jnp.inf
        invert_mask = invert_mask * jnp.inf
        # Set distance of all invalid areas to infinity
        valid_no_back_d = valid_no_back_d + invert_mask
        # jax.debug.print("valid_no_back_d2: {y}",y=valid_no_back_d)
        masked_dist = valid_no_back_d  # ghost_dist * valids * ghost_mask
        minimum_distance = jnp.min(masked_dist)
        # set all other values aside from the minimum to  0 and the minimum to 1
        masked_dist = jnp.where(masked_dist == minimum_distance, 1, 0)
        # jax.debug.print("masked_dist: {y}",y=masked_dist)
        # zero_mask = masked_dist != 0

        location_value = grid[new_player_pos[1]][new_player_pos[0]]
        # Check if colliding with walls
        false_cond = lambda x: False
        true_cond = lambda x: True
        collision = jax.lax.cond(location_value == 1, false_cond, true_cond, 0)
        # jax.debug.print("collision: {y}",y=collision)
        #jax.debug.print("valid_locs: {y}", y=valid_locs)
        #jax.debug.print("masked_dist: {y}", y=masked_dist)
        return collision, valid_locs, masked_dist

    def check_ghost_collisions(
        self, ghost_pos: chex.Array, new_player_pos: chex.Array, state: State
    ) -> Tuple[State, bool]:
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
        ) -> Tuple[int, State, bool]:

            eat = lambda: True
            no_eat = lambda: False
            frightened_time = state.frightened_state_time

            is_eat = jax.lax.cond(frightened_time > 0, eat, no_eat)

            ghost_p = Position(y=ghost_pos[0], x=ghost_pos[1])
            # jax.debug.print("ghost_p: {y}",y=ghost_p)

            cond_x = ghost_p.x == new_player_pos.x
            cond_y = ghost_p.y == new_player_pos.y
            cond = cond_x * cond_y

            ghost_reset = is_eat * cond
            ghost_init_steps = ghost_reset * 6

            def no_col_fn() -> Tuple[chex.Array, bool]:
                return ghost_pos, False

            def col_fn() -> Tuple[chex.Array, bool]:
                reset_true = lambda: (jnp.array(og_pos), False)
                reset_false = lambda: (ghost_pos, True)
                path, done = jax.lax.cond(ghost_reset, reset_true, reset_false)
                return path, done

            # First check for collision
            path, done = jax.lax.cond(cond, col_fn, no_col_fn)
            # jax.debug.print("dones: {y}",y=done)
            return path, ghost_init_steps, done

        ghost_positions, ghost_init, dones = jax.vmap(
            check_collisions, in_axes=(0, None, 0, None)
        )(ghost_pos, new_player_pos, jnp.array(og_pos), state)
        # jax.debug.print("ew_player_pos: {y}",y=new_player_pos)
        done = jnp.any(dones)
        jax.debug.print("done: {y}", y=done)
        state.ghost_locations = ghost_positions
        state.ghost_init_steps = jnp.array(ghost_init, int)

        return state, done

    def render(self, state: State) -> Any:
        """Render the given state of the environment.

        Args:
            state: `State` object containing the current environment state.
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
