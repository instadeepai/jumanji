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
from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.robot_warehouse.constants import _AGENTS, _SHELVES
from jumanji.environments.routing.robot_warehouse.types import Action, Agent, Entity
from jumanji.environments.routing.robot_warehouse.utils_agent import (
    get_agent_view,
    get_new_position_after_forward,
)
from jumanji.tree_utils import tree_slice


def get_entity_ids(entities: Entity) -> chex.Array:
    """Get ids for agents/shelves.

    Args:
        entities: a pytree of Agent or Shelf type.

    Returns:
        an array of ids.
    """
    return jnp.arange(entities[1].shape[0])


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
        return check_forward & (agent_id_at_target_pos > 0)

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
        shelf = jnp.array([1, requested], dtype=jnp.int32)
        obs, idx = write_to_observation(obs, idx, shelf)
        return obs, idx

    def agent_sensor_scan(
        obs_idx_and_agent_id: Tuple[chex.Array, chex.Array, chex.Array],
        agent_sensor: chex.Array,
    ) -> Tuple[Tuple[chex.Array, chex.Array, chex.Array], None]:
        """Write agent observation with agent sensor information
        of other agents.
        """
        obs, idx, agent_id = obs_idx_and_agent_id
        sensor_check_for_self = jnp.equal(agent_sensor, agent_id + 1)
        sensor_check_for_self_or_no_other = jnp.logical_or(
            jnp.equal(agent_sensor, 0),
            sensor_check_for_self,
        )
        obs, idx = jax.lax.cond(
            sensor_check_for_self_or_no_other,
            write_no_agent,
            write_agent,
            obs,
            idx,
            agent_sensor,
            sensor_check_for_self,
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
