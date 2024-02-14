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

from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from jax import nn

from jumanji.environments.routing.pac_man.types import Position, State


def ghost_move(
    state: State, x_size: int, y_size: int
) -> Tuple[chex.Array, chex.Array, chex.PRNGKey]:
    """
    Generate a move for each ghost and update their positions accordingly.

    Args:
        state: 'State` object corresponding to the new state of the environment.

    Returns:
        ghost_path: the new positions of each ghost after moving.
        key: new PRNG key.
    """

    key = state.key
    key, ghost_key_0, ghost_key_1, ghost_key_2, ghost_key_3 = jax.random.split(key, 5)
    ghost_keys = jnp.array([ghost_key_0, ghost_key_1, ghost_key_2, ghost_key_3])
    ghost_pos = state.ghost_locations
    player_pos = state.player_locations
    start_time = state.ghost_starts
    ghost_paths = []
    scatter_targets = state.scatter_targets
    actions = jnp.array(state.ghost_actions)

    def move(
        ghost_pos: chex.Array,
        tunnel_key: chex.PRNGKey,
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

        tunnel_key, ghost_tunnel_key = jax.random.split(tunnel_key)
        _, valids, valid_actions = check_ghost_wall_collisions(
            state,
            ghost_pos,
            ghost_num,
            player_pos,
            ghost_init_target,
            old_ghost_locations,
            scatter_target,
            x_size,
            y_size,
        )
        vert_col = jnp.array([1, 0, 1, 0])
        hor_col = jnp.array([0, 1, 0, 1])

        # If valids is  [1,0,1,0] or [0,1,0,1] then use old action
        # this is the case where the ghosts are in a tunnel as ghosts
        # are not allowed to backtrack.

        is_in_tunnel = jnp.logical_or(
            jnp.array_equal(valids, vert_col), jnp.array_equal(valids, hor_col)
        )

        def is_tunnel(
            inputs: Tuple[chex.Array, chex.Array, int, chex.PRNGKey, int]
        ) -> int:
            """Repeat old action if in tunnel"""
            _, _, ghost_action, _, _ = inputs
            return ghost_action

        def no_tunnel(
            inputs: Tuple[chex.Array, chex.Array, int, chex.PRNGKey, int]
        ) -> Any:
            """Chose new action when at intersection"""
            logits, actions, _, ghost_tunnel_key, _ = inputs
            _, ghost_tunnel_key = jax.random.split(ghost_tunnel_key)
            masked_logits = jnp.where(logits == 0, -1e9, logits)
            # Softmax distribution
            softmax_dist = nn.softmax(masked_logits)
            weighted_actions = actions * logits
            new_action = jax.random.choice(
                ghost_tunnel_key, weighted_actions, p=softmax_dist
            ).astype(int)
            return new_action

        inputs_no_tunnel = (
            valid_actions,
            act,
            ghost_action,
            ghost_tunnel_key,
            ghost_num,
        )

        def start_over(
            inputs: Tuple[chex.Array, chex.Array, int, chex.PRNGKey, int]
        ) -> Any:
            """If not in waiting mode then pick new action"""

            chosen_action = jax.lax.cond(is_in_tunnel, is_tunnel, no_tunnel, inputs)
            return jnp.squeeze(chosen_action)

        def no_start(
            inputs: Tuple[chex.Array, chex.Array, int, chex.PRNGKey, int]
        ) -> int:
            """If in waiting mode then use no-op"""
            return 4

        position = ghost_pos
        chosen_action = jax.lax.cond(
            ghost_start < 0, start_over, no_start, inputs_no_tunnel
        )

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

        # If using the teleporter shift to the other side of the map
        path = jnp.array([new_pos_col % y_size, new_pos_row % x_size])
        path = jax.lax.cond(ghost_start <= 0, lambda: path, lambda: position)

        return path, chosen_action

    ghost_nums = jnp.array([0, 1, 2, 3])
    ghost_init_targets = state.ghost_init_targets
    ghost_paths, ghost_acts = jax.vmap(move, in_axes=(0, 0, 0, 0, None, 0, 0, 0, 0))(
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


def check_ghost_wall_collisions(
    state: State,
    new_player_pos: chex.Array,
    ghost_num: int,
    pacman_pos: Position,
    init_target: chex.Array,
    old_ghost_locations: chex.Array,
    scatter_target: chex.Array,
    x_size: int,
    y_size: int,
) -> Tuple[bool, chex.Array, chex.Array]:

    """
    Determine valid move for the ghost.

    Args:
        state: 'State` object corresponding to the new state of the environment.
        new_player_pos: the position of the player after the last action.
        ghost_num: the id of the ghost being moved.
        pacman_pos: the current position of the player.
        init_target: number of steps left of initial movement.
        old_ghost_locations: the position of the ghost in the previous step.
        scatter_target: the scatter target of the ghost being moved.

    Returns:
        collision: a boolean indicating if the player has moved into a wall.
        valid_locs: jax array of all possible valid locations.
        masked_dist: jax array of valid target locations.
    """

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

    # For ghost 0: Move to closest tile to pacman
    def red_ghost(pacman_pos: Position) -> Tuple[chex.Array, chex.Array]:
        """
        Select targets for red ghost as distance from current tile occupied by
        pacman.
        """

        distance_list = jax.vmap(get_directions, in_axes=(None, 0))(pacman_pos, ghost_p)
        ghost_dist = jax.vmap(jnp.linalg.norm, in_axes=(0))(distance_list)
        return distance_list, ghost_dist

    # For ghost 1: move 4 steps ahead of pacman
    def pink_ghost(
        pacman_pos: Position, steps: int = 4
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Select targets for pink ghost as distance from the tile 4 steps ahead
        of the current position of pacman.
        """

        pac_pos = player_step(state, pac_dir, x_size, y_size, steps=steps)
        distance_list = jax.vmap(get_directions, in_axes=(None, 0))(pac_pos, ghost_p)
        ghost_dist = jax.vmap(jnp.linalg.norm, in_axes=(0))(distance_list)

        return distance_list, ghost_dist

    # For ghost 2: move to averge between 0 and 1
    def blue_ghost(pacman_pos: Position) -> Tuple[chex.Array, chex.Array]:
        """
        Select targets for blue ghost as sum of the targets of the pink and red
        ghost.
        """

        ghost_0_target, _ = red_ghost(pacman_pos)
        ghost_1_target, _ = pink_ghost(pacman_pos=pacman_pos, steps=4)
        distance_list = ghost_0_target + ghost_1_target
        ghost_dist = jax.vmap(jnp.linalg.norm, in_axes=(0))(distance_list)
        return distance_list, ghost_dist

    def orange_ghost(pacman_pos: Position) -> Tuple[chex.Array, chex.Array]:
        """
        Select targets for orange ghost as distance from the current tile
        occupied by pacman if greater than 8 units of distance from pacman.
        If closed than 8 units then the target is the scatter target tile.
        """

        distance_pacman = jnp.array(
            [new_player_pos[0] - pacman_pos.x, new_player_pos[1] - pacman_pos.y]
        )
        distance_pacman = jnp.linalg.norm(distance_pacman)

        distance_list = jax.vmap(get_directions, in_axes=(None, 0))(pacman_pos, ghost_p)

        _, ghost_dist = jax.lax.cond(
            distance_pacman > 8, red_ghost, scared_behaviors, pacman_pos
        )

        return distance_list, ghost_dist

    def general_behaviors(pacman_pos: Position) -> Tuple[chex.Array, chex.Array]:
        """
        General behaviors for ghosts
        based on https://gameinternals.com/understanding-pac-man-ghost-behavior
        """
        blinky = lambda pacman_pos: red_ghost(pacman_pos)
        inky = lambda pacman_pos: pink_ghost(pacman_pos)
        pinky = lambda pacman_pos: blue_ghost(pacman_pos)
        clyde = lambda pacman_pos: orange_ghost(pacman_pos)
        _, ghost_dist = jax.lax.switch(
            ghost_num, [blinky, inky, pinky, clyde], pacman_pos
        )
        return _, ghost_dist

    def scared_behaviors(pacman_pos: Position) -> Tuple[chex.Array, chex.Array]:
        """Select scatter targets if power pellet active"""
        scatter_pos = Position(x=scatter_target[1], y=scatter_target[0])
        _, ghost_dist = red_ghost(scatter_pos)
        return _, ghost_dist

    def behaviors() -> Tuple[chex.Array, chex.Array]:
        """Select scatter or normal targets"""
        _, ghost_dist = jax.lax.cond(
            is_scared > 0, scared_behaviors, general_behaviors, pacman_pos
        )
        return _, ghost_dist

    def init_behaviors() -> Tuple[chex.Array, chex.Array]:
        """Set initial fixed behaviors"""
        target = Position(y=init_target[1], x=init_target[0])
        _, ghost_dist = red_ghost(pacman_pos=target)
        return _, ghost_dist

    _, ghost_dist = jax.lax.cond(
        ghost_init_steps[ghost_num] > 0, init_behaviors, behaviors
    )

    def get_valid_positions(pos: chex.Array) -> Any:
        """Get values of surrounding positions"""
        value = grid[pos[0]][pos[1]]
        return value

    valids = jax.vmap(get_valid_positions, in_axes=(0,))(ghost_p)
    valid_locs = valids

    # Set all invalid areas to an infinite cost
    # remove back tracking
    valid_no_back = valids * ghost_mask
    # Get distances of valid locations
    valid_no_back_d = valid_no_back * ghost_dist
    invert_mask = valid_no_back != 1
    invert_mask = invert_mask * jnp.inf
    # Set distance of all invalid areas to infinity
    valid_no_back_d = valid_no_back_d + invert_mask
    masked_dist = valid_no_back_d
    minimum_distance = jnp.min(masked_dist)
    # set all other values aside from the minimum to  0 and the minimum to 1
    masked_dist = jnp.where(masked_dist == minimum_distance, 1, 0)

    location_value = grid[new_player_pos[1]][new_player_pos[0]]
    # Check if colliding with walls
    collision = location_value != 1
    return collision, valid_locs, masked_dist


def check_ghost_collisions(
    ghost_pos: chex.Array, new_player_pos: chex.Array, state: State
) -> Tuple[State, bool, chex.Numeric]:
    """
    Check if the player collides with a ghost.

    Args:
        ghost_pos: A list of tuples representing the current positions of the ghosts.
        new_player_pos: The current position of the player.
        state: 'State` object corresponding to the new state of the environment.

    Returns:
        state: 'State` object corresponding to the new state of the environment.
        done: check for game over from collision with ghost.
        col_rewards: reward for eating ghost.
    """

    og_pos = state.initial_ghost_positions
    ghost_eaten = state.ghost_eaten

    def check_collisions(
        ghost_pos: chex.Array,
        new_player_pos: chex.Array,
        og_pos: chex.Array,
        state: State,
        ghost_eaten: chex.Numeric,
        old_ghost_pos: chex.Array,
    ) -> Tuple[chex.Array, chex.Numeric, chex.Numeric, chex.Numeric, chex.Numeric]:
        """Check if ghost has collided with player"""
        frightened_time = state.frightened_state_time
        is_eat = frightened_time > 0

        ghost_p = Position(y=ghost_pos[0], x=ghost_pos[1])
        old_ghost_p = Position(y=old_ghost_pos[0], x=old_ghost_pos[1])

        # Check if new player pos is the same as the old ghost pos
        cond_x1 = ghost_p.x == new_player_pos.x
        cond_y1 = ghost_p.y == new_player_pos.y
        cond1 = cond_x1 * cond_y1

        # Check if new ghost position is the old player pos
        cond_x2 = ghost_p.x == state.player_locations.x
        cond_y2 = ghost_p.y == state.player_locations.y
        cond2 = cond_x2 * cond_y2

        # Check if new player pos is the same as old ghost pos
        cond_x3 = old_ghost_p.x == new_player_pos.x
        cond_y3 = old_ghost_p.y == new_player_pos.y

        cond3 = cond_x3 * cond_y3
        cond = cond1 | cond2 | cond3

        ghost_reset = is_eat * cond
        ghost_init_steps = ghost_reset * 6
        edible = ghost_eaten

        def no_col_fn() -> Tuple[chex.Array, chex.Numeric, chex.Numeric, chex.Numeric]:
            return ghost_pos, False, 0.0, edible

        def col_fn() -> Tuple[chex.Array, chex.Numeric, chex.Numeric, chex.Numeric]:
            reset_true = lambda: (jnp.array(og_pos), False, 200.0, False)
            reset_false = lambda: (ghost_pos, True, 0.0, edible)
            path, done, col_reward, ghost_eaten = jax.lax.cond(
                ghost_reset, reset_true, reset_false
            )
            return path, done, col_reward, ghost_eaten

        # First check for collision
        path, done, col_reward, ghost_eaten = jax.lax.cond(cond, col_fn, no_col_fn)
        col_reward = col_reward * edible
        return path, ghost_init_steps, done, col_reward, ghost_eaten

    old_ghost_positions = state.old_ghost_locations
    ghost_positions, ghost_init, dones, col_rewards, ghost_eaten = jax.vmap(
        check_collisions, in_axes=(0, None, 0, None, 0, 0)
    )(
        ghost_pos,
        new_player_pos,
        jnp.array(og_pos),
        state,
        ghost_eaten,
        old_ghost_positions,
    )
    done = jnp.any(dones)
    col_rewards = col_rewards.sum()
    state.ghost_locations = ghost_positions
    state.ghost_eaten = ghost_eaten

    return state, done, col_rewards


def get_directions(pacman_position: Position, ghost_position: chex.Array) -> chex.Array:
    """Get the vector distance between the ghost and the target position"""
    direction = jnp.array(
        [
            ghost_position[0] - pacman_position.y,
            ghost_position[1] - pacman_position.x,
        ]
    )
    return direction


def player_step(
    state: State, action: int, x_size: int, y_size: int, steps: int = 1
) -> Position:
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

    new_pos = Position(x=new_pos_col % x_size, y=new_pos_row % y_size)
    return new_pos
