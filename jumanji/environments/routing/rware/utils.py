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

import functools
from typing import Tuple, Union

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.rware.types import (
    _AGENTS,
    _SHELVES,
    Action,
    Agent,
    Direction,
    Entity,
    Position,
    Shelf,
)
from jumanji.tree_utils import tree_add_element, tree_slice


def get_entity_ids(entities: Entity) -> chex.Array:
    """Get ids for agents/shelves.

    Args:
        entities: a pytree of Agent or Shelf type.

    Returns:
        an array of ids.
    """
    return jnp.arange(entities[1].shape[0])


def spawn_agent(
    agent_coordinates: chex.Array,
    direction: chex.Array,
) -> chex.Array:
    """Spawn an agent (robot) at a random position and in a random direction.

    Args:
        key: pseudo random number key.

    Returns:
        spawned agent."""
    x, y = agent_coordinates
    agent_pos = Position(x=x, y=y)
    agent = Agent(position=agent_pos, direction=direction, is_carrying=0)
    return agent


def spawn_shelf(
    shelf_coordinates: chex.Array,
    requested: chex.Array,
) -> chex.Array:
    """Spawn a shelf at a specific shelf position and label the shelf
    as requested or not.

    Args:
        shelf_id: id of the shelf being spawned.
        requested: whether the shelf has been requested or not.

    Returns:
        spawned shelf."""
    x, y = shelf_coordinates
    shelf_pos = Position(x=x, y=y)
    shelf = Shelf(position=shelf_pos, is_requested=requested)
    return shelf


def spawn_random_entities(
    key: chex.PRNGKey,
    grid_size: chex.Array,
    agent_ids: chex.Array,
    shelf_ids: chex.Array,
    shelf_coordinates: chex.Array,
    request_queue_size: chex.Array,
) -> Tuple[chex.PRNGKey, Agent, Shelf, chex.Array]:
    """Spawn agents and shelves on the warehouse floor grid.

    Args:
        key: pseudo random number key.
        grid_size: the size of the warehouse floor grid.
        agent_ids: array of agent ids.
        shelf_ids: array of shelf ids.
        shelf_coordinates: x,y coordinates of shelf positions.
        request_queue_size: the number of shelves to be delivered.

    Returns:
        new key, spawned agents, shelves and the request queue.
    """

    # random agent positions
    num_agents = len(agent_ids)
    key, position_key = jax.random.split(key)
    grid_cells = jnp.array(jnp.arange(grid_size[0] * grid_size[1]))
    agent_coords = jax.random.choice(
        position_key,
        grid_cells,
        shape=(num_agents,),
        replace=False,
    )
    agent_coords = jnp.transpose(
        jnp.asarray(jnp.unravel_index(agent_coords, grid_size))
    )

    # random agent directions
    key, direction_key = jax.random.split(key)
    possible_directions = jnp.array([d.value for d in Direction])
    agent_dirs = jax.random.choice(
        direction_key, possible_directions, shape=(num_agents,)
    )

    # sample request queue
    key, queue_key = jax.random.split(key)
    shelf_request_queue = jax.random.choice(
        queue_key,
        shelf_ids,
        shape=(request_queue_size,),
        replace=False,
    )
    requested_ids = jnp.zeros(shelf_ids.shape)
    requested_ids = requested_ids.at[shelf_request_queue].set(1)

    # spawn agents and shelves
    agents = jax.vmap(spawn_agent)(agent_coords, agent_dirs)
    shelves = jax.vmap(spawn_shelf)(shelf_coordinates, requested_ids)
    return key, agents, shelves, shelf_request_queue


def place_entity_on_grid(
    grid: chex.Array,
    channel: chex.Array,
    entities: Entity,
    entity_id: chex.Array,
) -> chex.Array:
    """Places an entity (Agent/Shelf) on the grid based on its
    (x, y) position defined once spawned.

    Args:
        grid: the warehouse floor grid array.
        channel: the grid channel index, either agents or shelves.
        entities: a pytree of Agent or Shelf type containing entity information.
        entity_id: unique ID identifying a specific entity.

    Returns:
        the warehouse grid with the specific entity in its position.
    """
    entity = tree_slice(entities, entity_id)
    x, y = entity.position.x, entity.position.y
    grid = grid.at[channel, x, y].set(entity_id + 1)
    return grid


def place_entities_on_grid(
    grid: chex.Array, agents: Agent, shelves: Shelf
) -> chex.Array:
    """Place agents and shelves on the grid.

    Args:
        grid: the warehouse floor grid array.
        agents: a pytree of Agent type containing agent information.
        shelves: a pytree of Shelf type containing shelf information.

    Returns:
        the warehouse grid with all agents and shelves placed in their
        positions.
    """
    agent_ids = get_entity_ids(agents)
    shelf_ids = get_entity_ids(shelves)

    # place agents and shelves on warehouse grid
    def place_agents_scan(
        grid_and_agents: Tuple[chex.Array, chex.Array], agent_id: chex.Array
    ) -> Tuple[Tuple[chex.Array, chex.Array], None]:
        grid, agents = grid_and_agents
        grid = place_entity_on_grid(grid, _AGENTS, agents, agent_id)
        return (grid, agents), None

    def place_shelves_scan(
        grid_and_shelves: Tuple[chex.Array, chex.Array], shelf_id: chex.Array
    ) -> Tuple[Tuple[chex.Array, chex.Array], None]:
        grid, shelves = grid_and_shelves
        grid = place_entity_on_grid(grid, _SHELVES, shelves, shelf_id)
        return (grid, shelves), None

    (grid, _), _ = jax.lax.scan(place_agents_scan, (grid, agents), agent_ids)
    (grid, _), _ = jax.lax.scan(place_shelves_scan, (grid, shelves), shelf_ids)
    return grid


def update_agent(
    agents: Agent,
    agent_id: chex.Array,
    attr: str,
    value: Union[chex.Array, Position],
) -> Agent:
    """Update the attribute information of a specific agent.

    Args:
        agents: a pytree of either Agent type containing agent information.
        agent_id: unique ID identifying a specific agent.
        attr: the attribute to update, e.g. `direction`, or `is_requested`.
        value: the new value to which the attribute is to be set.

    Returns:
        the agent with the specified attribute updated to the given value.
    """
    params = {attr: value}
    agent = tree_slice(agents, agent_id)
    agent = agent._replace(**params)
    agents: Agent = tree_add_element(agents, agent_id, agent)
    return agents


def update_shelf(
    shelves: Shelf,
    shelf_id: chex.Array,
    attr: str,
    value: Union[chex.Array, Position],
) -> Shelf:
    """Update the attribute information of a specific shelf.

    Args:
        shelves: a pytree of Shelf type containing shelf information.
        shelf_id: unique ID identifying a specific shelf.
        attr: the attribute to update, e.g. `direction`, or `is_requested`.
        value: the new value to which the attribute is to be set.

    Returns:
        the shelf with the specified attribute updated to the given value.
    """
    params = {attr: value}
    shelf = tree_slice(shelves, shelf_id)
    shelf = shelf._replace(**params)
    shelves: Shelf = tree_add_element(shelves, shelf_id, shelf)
    return shelves


def get_new_direction_after_turn(
    action: chex.Array, agent_direction: chex.Array
) -> chex.Array:
    """Get the correct direction the agent should face given
    the turn action it took. E.g. if the agent is facing LEFT
    and turns RIGHT it should now be facing UP, etc.

    Args:
        action: the agent's action.
        agent_direction: the agent's current direction.

    Returns:
        the direction the agent should be facing given the action it took.
    """
    change_in_direction = jnp.array([0, 0, -1, 1, 0])[action]
    new_agent_direction = (agent_direction + change_in_direction) % 4
    return new_agent_direction


def get_new_position_after_forward(
    grid: chex.Array, agent_position: chex.Array, agent_direction: chex.Array
) -> Position:
    """Get the correct position the agent will be in after moving forward
    in its current direction. E.g. if the agent is facing LEFT and turns
    RIGHT it should stay in the same position. If instead it moves FORWARD
    it should move left by one cell.

    Args:
        grid: the warehouse floor grid array.
        agent_position: the agent's current position.
        agent_direction: the agent's current direction.

    Returns:
        the position the agent should be in given the action it took.
    """
    _, grid_width, grid_height = grid.shape
    x, y = agent_position.x, agent_position.y
    move_up = lambda x, y: Position(jnp.max(jnp.array([0, x - 1])), y)
    move_right = lambda x, y: Position(x, jnp.min(jnp.array([grid_height - 1, y + 1])))
    move_down = lambda x, y: Position(jnp.min(jnp.array([grid_width - 1, x + 1])), y)
    move_left = lambda x, y: Position(x, jnp.max(jnp.array([0, y - 1])))
    new_position: Position = jax.lax.switch(
        agent_direction, [move_up, move_right, move_down, move_left], x, y
    )
    return new_position


def is_valid_action(grid: chex.Array, agent: Agent, action: chex.Array) -> chex.Array:
    """If the agent is carrying a shelf and collides with another
    shelf based on its current action, this action is deemed invalid.

    Args:
        grid: the warehouse floor grid array.
        agent: the agent for which the action is being checked.
        action: the action the agent is about to take.

    Returns:
        a boolean indicating whether the action is valid or not.
    """

    # get start and target positions
    start = agent.position
    target = get_new_position_after_forward(grid, start, agent.direction)

    # check if carrying and walking into another shelf
    cond = jnp.logical_and(jnp.equal(action, Action.FORWARD), agent.is_carrying)
    cond = jnp.logical_and(cond, jnp.logical_not(jnp.array_equal(start, target)))
    cond = jnp.logical_and(cond, grid[_SHELVES, target.x, target.y])

    return ~cond


def get_valid_actions(actions: chex.Array, action_mask: chex.Array) -> chex.Array:
    """Get the valid action the agent should take given its action mask.

    Args:
        actions: the actions the agents are about to take.
        action_mask: the mask of valid actions.

    Returns:
        the action the agent should take given its current state.
    """

    def get_valid_action(action_mask: chex.Array, action: chex.Array) -> chex.Array:
        return jax.lax.cond(action_mask[action], lambda: action, lambda: 0)

    return jax.vmap(get_valid_action)(action_mask, actions)


def is_collision(grid: chex.Array, agent: Agent, action: int) -> chex.Array:
    """Calculate whether an agent is about to collide with another
    entity. If the agent instead collides with another agent, the
    episode terminates (this behavior is specific to this JAX version).

    Args:
        grid: the warehouse floor grid array.
        agent: the agent for which collisions are being checked.

    Returns:
        a boolean indicating whether the agent collided with another
        agent or not.
    """

    def check_collision() -> chex.Array:
        # get start and target positions
        start = agent.position
        target = get_new_position_after_forward(grid, start, agent.direction)

        agent_id_at_target_pos = grid[_AGENTS, target.x, target.y]
        check_forward = ~jnp.array_equal(start, target)
        collision = check_forward & (agent_id_at_target_pos > 0)
        return collision

    return jax.lax.cond(
        jnp.equal(action, Action.FORWARD), check_collision, lambda: False
    )


def calculate_num_observation_features(sensor_range: chex.Array) -> chex.Array:
    """Calculates the 1-d size of the agent observations array based on the
    environment parameters at instantiation

    Below is a receptive field for an agent x with a sensor range of 1:

                            O O O
                            O x O
                            O O O

    For the sensor on the agent's own position,
    we have the following features
    1. the agent's position -> dim 2
    2. is the agent carrying a shelf? -> binary {0, 1} with dim 1
    3. the direction of the agent -> one-hot with dim 4
    4. is the agent on the warehouse "highway" or not? -> binary with dim 1
    Total dim for agent in focus = 2 + 1 + 4 + 1 = 8

    Then, for each sensor position (other than the agent's own position, 8 in total),
    we have the following features based on other agents:
    1. is there an agent? -> binary {0, 1} with dim 1
    2. if yes, the agent's direction -> one-hot with dim 4, if no, fill all zeros
    Therefore, the total number of dimensions for other agent features
    (1 + 4) * num_obs_sensors

    Finally, for each sensor position (9 in total) in the agent's receptive field,
    we have the following features based on shelves:
    1. is there a shelf? -> binary {0, 1} with dim 1
    2. if so, has this shelf been requested -> binary {0, 1} with dim 1, if no, zero
    Therefore, the total number of dimensions for shelf features is
    (1 + 1) * num_obs_sensors

    Args:
        sensor_range: the range of the agent's sensors.

    Returns:
        agent's 1-d observation array.
    """
    num_obs_sensors = (1 + 2 * sensor_range) ** 2
    obs_features = 8  # agent's own features
    obs_features += (num_obs_sensors - 1) * 5  # other agent features
    obs_features += num_obs_sensors * 2  # shelf features
    return jnp.array(obs_features, jnp.int32)


def write_to_observation(
    observation: chex.Array, idx: chex.Array, data: chex.Array
) -> Tuple[chex.Array, chex.Array]:
    """Write data to the given observation vector at a specified index

    Args:
        observation: an observation to which data will be written.
        idx: an integer representing the index at which the data will be inserted.
        data: the data that will be inserted into the observation array.

    Returns:
        the updated observation array and the new index
        of where to insert the next data.
    """
    data_size = len(data)
    observation = jax.lax.dynamic_update_slice(observation, data, (idx,))
    return observation, idx + data_size


def move_writer_index(idx: chex.Array, bits: chex.Array) -> chex.Array:
    """Skip an indicated number of bits in the observation array being written.

    Args:
        idx: an integer representing the index at which to skip bits.
        bits: the number of bits to skip.

    Returns:
        the new index at which to insert data.
    """
    return idx + bits


def get_agent_view(
    grid: chex.Array, agent: chex.Array, sensor_range: chex.Array
) -> Tuple[chex.Array, chex.Array]:
    """Get an agent's view of other agents and shelves within its
    sensor range.

    Below is an example of the agent's view of other agents from
    the perspective of agent 1 with a sensor range of 1:

                            0, 0, 0
                            0, 1, 2
                            0, 0, 0

    It sees agent 2 to its right. Separately, the view of shelves
    is shown below:

                            0, 0, 0
                            0, 3, 4
                            0, 7, 8

    Agent 1 is on top of shelf 3 and has 4, 7 and 8 around it in
    the bottom right corner of its view. Before returning these
    views they are flattened into a 1-d arrays, i.e.

    View of agents: [0, 0, 0, 0, 1, 2, 0, 0, 0]
    View of shelves: [0, 0, 0, 0, 3, 4, 0, 7, 8]


    Args:
        grid: the warehouse floor grid array.
        agent: the agent for which the view of their receptive field
            is to be calculated.
        sensor_range: the range of the agent's sensors.

    Returns:
        a view of the agents receptive field separated into two arrays:
        one for other agents and one for shelves.
    """
    receptive_field = sensor_range * 2 + 1
    padded_agents_layer = jnp.pad(grid[_AGENTS], sensor_range, mode="constant")
    padded_shelves_layer = jnp.pad(grid[_SHELVES], sensor_range, mode="constant")
    agent_view_of_agents = jax.lax.dynamic_slice(
        padded_agents_layer,
        (agent.position.x, agent.position.y),
        (receptive_field, receptive_field),
    ).reshape(-1)
    agent_view_of_shelves = jax.lax.dynamic_slice(
        padded_shelves_layer,
        (agent.position.x, agent.position.y),
        (receptive_field, receptive_field),
    ).reshape(-1)
    return agent_view_of_agents, agent_view_of_shelves


def make_agent_observation(
    grid: chex.Array,
    agents: chex.Array,
    shelves: chex.Array,
    sensor_range: int,
    num_obs_features: int,
    highways: chex.Array,
    agent_id: int,
) -> chex.Array:
    """Create an observation for a single agent based on its view
    of other agents and shelves.

    Args:
        grid: the warehouse floor grid array.
        agents: a pytree of either Agent type containing agent information.
        shelves: a pytree of Shelf type containing shelf information.
        sensor_range: the range of the agent's sensors.
        num_obs_features: the number of features in the observation array.
        highways: binary array indicating highway positions.
        agent_id: unique ID identifying a specific agent.

    Returns:
        a 1-d array containing the agent's observation.
    """
    agent = tree_slice(agents, agent_id)
    agents_grid, shelves_grid = get_agent_view(grid, agent, sensor_range)

    # write flattened observations
    obs = jnp.zeros(num_obs_features, dtype=jnp.int32)
    idx = 0

    # write current agent position and whether carrying a shelf or not
    obs, idx = write_to_observation(
        obs,
        idx,
        jnp.array(
            [agent.position.x, agent.position.y, agent.is_carrying],
            dtype=jnp.int32,
        ),
    )

    # write current agent direction
    direction = jax.nn.one_hot(agent.direction, 4, dtype=jnp.int32)
    obs, idx = write_to_observation(obs, idx, direction)

    # write if agent is on highway or not
    obs, idx = write_to_observation(
        obs,
        idx,
        jnp.array(
            [jnp.array(highways[agent.position.x, agent.position.y], int)],
            dtype=jnp.int32,
        ),
    )

    # function for writing receptive field cells
    def write_no_agent(
        obs: chex.Array, idx: int, _: int, is_self: bool
    ) -> Tuple[chex.Array, int]:
        "Write information for empty agent cell."
        # if there is no agent we set a 0 and all zeros
        # for the direction as well, i.e. [0, 0, 0, 0, 0]
        idx = jax.lax.cond(is_self, lambda i: i, lambda i: move_writer_index(i, 5), idx)
        return obs, idx

    def write_agent(
        obs: chex.Array, idx: int, id_agent: int, _: bool
    ) -> Tuple[chex.Array, int]:
        "Write information for cell containing an agent."
        obs, idx = write_to_observation(obs, idx, jnp.array([1], dtype=jnp.int32))
        direction = jax.nn.one_hot(
            tree_slice(agents, id_agent - 1).direction, 4, dtype=jnp.int32
        )
        obs, idx = write_to_observation(obs, idx, direction)
        return obs, idx

    def write_no_shelf(obs: chex.Array, idx: int, _: int) -> Tuple[chex.Array, int]:
        "write information for empty shelf cell."
        idx = move_writer_index(idx, 2)
        return obs, idx

    def write_shelf(obs: chex.Array, idx: int, shelf_id: int) -> Tuple[chex.Array, int]:
        "Write information for cell containing a shelf."
        requested = tree_slice(shelves, shelf_id - 1).is_requested
        obs, idx = write_to_observation(
            obs,
            idx,
            jnp.array(
                [1, requested],
                dtype=jnp.int32,
            ),
        )
        return obs, idx

    def agent_sensor_scan(
        obs_idx_and_agent_id: Tuple[chex.Array, chex.Array, chex.Array],
        agent_sensor: chex.Array,
    ) -> Tuple[Tuple[chex.Array, chex.Array, chex.Array], None]:
        """Write agent observation with agent sensor information
        of other agents.
        """
        obs, idx, agent_id = obs_idx_and_agent_id
        cond1 = jnp.equal(agent_sensor, agent_id + 1)
        cond2 = jnp.logical_or(
            jnp.equal(agent_sensor, 0),
            cond1,
        )
        obs, idx = jax.lax.cond(
            cond2,
            write_no_agent,
            write_agent,
            obs,
            idx,
            agent_sensor,
            cond1,
        )
        return (obs, idx, agent_id), None

    def shelf_sensor_scan(
        obs_and_idx: Tuple[chex.Array, chex.Array], shelf_sensor: chex.Array
    ) -> Tuple[Tuple[chex.Array, chex.Array], None]:
        """Write agent observation with agent sensor information
        of other shelves.
        """
        obs, idx = obs_and_idx
        obs, idx = jax.lax.cond(
            jnp.equal(shelf_sensor, 0),
            write_no_shelf,
            write_shelf,
            obs,
            idx,
            shelf_sensor,
        )
        return (obs, idx), None

    (obs, idx, _), _ = jax.lax.scan(
        agent_sensor_scan, (obs, idx, agent_id), agents_grid
    )
    (obs, _), _ = jax.lax.scan(shelf_sensor_scan, (obs, idx), shelves_grid)
    return obs


def set_agent_carrying_if_at_shelf_position(
    grid: chex.Array, agents: chex.Array, agent_id: int, is_highway: chex.Array
) -> chex.Array:
    """Set the agent as carrying a shelf if it is at a shelf position.

    Args:
        grid: the warehouse floor grid array.
        agents: a pytree of either Agent type containing agent information.
        agent_id: unique ID identifying a specific agent.
        is_highway: binary value indicating highway position.

    Returns:
        updated agents pytree.
    """
    agent = tree_slice(agents, agent_id)
    shelf_id = grid[_SHELVES, agent.position.x, agent.position.y]

    return jax.lax.cond(
        shelf_id > 0,
        lambda: update_agent(agents, agent_id, "is_carrying", 1),
        lambda: agents,
    )


def offload_shelf_if_position_is_open(
    grid: chex.Array, agents: chex.Array, agent_id: int, is_highway: chex.Array
) -> chex.Array:
    """Set the agent as not carrying a shelf if it is at a shelf position.

    Args:
        grid: the warehouse floor grid array.
        agents: a pytree of either Agent type containing agent information.
        agent_id: unique ID identifying a specific agent.
        is_highway: binary value indicating highway position.

    Returns:
        updated agents pytree.
    """
    return jax.lax.cond(
        jnp.logical_not(is_highway),
        lambda: update_agent(agents, agent_id, "is_carrying", 0),
        lambda: agents,
    )


def set_carrying_shelf_if_load_toggled_and_not_carrying(
    grid: chex.Array,
    agents: chex.Array,
    action: int,
    agent_id: int,
    is_highway: chex.Array,
) -> chex.Array:
    """Set the agent as carrying a shelf if the load toggle action is
    performed and the agent is not carrying a shelf.

    Args:
        grid: the warehouse floor grid array.
        agents: a pytree of either Agent type containing agent information.
        action: the agent's action.
        agent_id: unique ID identifying a specific agent.
        is_highway: binary value indicating highway position.

    Returns:
        updated agents pytree.
    """
    agent = tree_slice(agents, agent_id)

    agents = jax.lax.cond(
        (action == Action.TOGGLE_LOAD.value) & ~agent.is_carrying,
        set_agent_carrying_if_at_shelf_position,
        offload_shelf_if_position_is_open,
        grid,
        agents,
        agent_id,
        is_highway,
    )
    return agents


def rotate_agent(
    grid: chex.Array,
    agents: chex.Array,
    action: int,
    agent_id: int,
    is_highway: chex.Array,
) -> chex.Array:
    """Rotate the agent in the direction of the action.

    Args:
        grid: the warehouse floor grid array.
        agents: a pytree of either Agent type containing agent information.
        action: the agent's action.
        agent_id: unique ID identifying a specific agent.
        is_highway: binary value indicating highway position.

    Returns:
        updated agents pytree.
    """
    agent = tree_slice(agents, agent_id)
    new_direction = get_new_direction_after_turn(action, agent.direction)
    return update_agent(agents, agent_id, "direction", new_direction)


def set_new_shelf_position_if_carrying(
    grid: chex.Array,
    shelves: chex.Array,
    cur_pos: chex.Array,
    new_pos: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Set the new position of the shelf if the agent is carrying one.

    Args:
        grid: the warehouse floor grid array.
        shelves: a pytree of Shelf type containing shelf information.
        cur_pos: the current position of the shelf.
        new_pos: the new position of the shelf.

    Returns:
        updated grid array and shelves pytree.
    """
    # update shelf position
    shelf_id = grid[_SHELVES, cur_pos.x, cur_pos.y]
    shelves = update_shelf(shelves, shelf_id - 1, "position", new_pos)

    # update shelf grid placement
    grid = grid.at[_SHELVES, cur_pos.x, cur_pos.y].set(0)
    grid = grid.at[_SHELVES, new_pos.x, new_pos.y].set(shelf_id)
    return grid, shelves


def set_new_position_after_forward(
    grid: chex.Array,
    agents: chex.Array,
    shelves: chex.Array,
    action: int,
    agent_id: int,
    is_highway: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Set the new position of the agent after a forward action.

    Args:
        grid: the warehouse floor grid array.
        agents: a pytree of either Agent type containing agent information.
        shelves: a pytree of Shelf type containing shelf information.
        action: the agent's action.
        agent_id: unique ID identifying a specific agent.
        is_highway: binary value indicating highway position.

    Returns:
        updated grid array, agents and shelves pytrees.
    """
    # update agent position
    agent = tree_slice(agents, agent_id)
    current_position = agent.position
    new_position = get_new_position_after_forward(grid, agent.position, agent.direction)
    agents = update_agent(agents, agent_id, "position", new_position)

    # update agent grid placement
    grid = grid.at[_AGENTS, current_position.x, current_position.y].set(0)
    grid = grid.at[_AGENTS, new_position.x, new_position.y].set(agent_id + 1)

    grid, shelves = jax.lax.cond(
        agent.is_carrying,
        set_new_shelf_position_if_carrying,
        lambda g, s, p, np: (g, s),
        grid,
        shelves,
        current_position,
        new_position,
    )
    return grid, agents, shelves


def set_new_direction_after_turn(
    grid: chex.Array,
    agents: chex.Array,
    shelves: chex.Array,
    action: int,
    agent_id: int,
    is_highway: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Set the new direction of the agent after a turning action.

    Args:
        grid: the warehouse floor grid array.
        agents: a pytree of either Agent type containing agent information.
        shelves: a pytree of Shelf type containing shelf information.
        action: the agent's action.
        agent_id: unique ID identifying a specific agent.
        is_highway: binary value indicating highway position.

    Returns:
        updated grid array, agents and shelves pytrees.
    """
    agents = jax.lax.cond(
        jnp.isin(action, jnp.array([Action.LEFT.value, Action.RIGHT.value])),
        rotate_agent,
        set_carrying_shelf_if_load_toggled_and_not_carrying,
        grid,
        agents,
        action,
        agent_id,
        is_highway,
    )
    return grid, agents, shelves


def compute_action_mask(grid: chex.Array, agents: Agent) -> chex.Array:
    """Compute the action mask for the environment.

    Args:
        grid: the warehouse floor grid array.
        agents: a pytree of either Agent type containing agent information.

    Returns:
        the action mask for the environment.
    """
    # vmap over agents and possible actions
    action_mask = jax.vmap(
        jax.vmap(functools.partial(is_valid_action, grid), in_axes=(None, 0)),
        in_axes=(0, None),
    )(agents, jnp.arange(5))
    return action_mask
