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

import chex
import jax
import jax.numpy as jnp
import pytest
from jumanji.tree_utils import tree_slice

from zathura.environments.routing.rware.types import (
    Action,
    Agent,
    Position,
    Shelf,
    State,
)
from zathura.environments.routing.rware.utils import (
    calculate_num_observation_features,
    compute_action_mask,
    get_agent_view,
    get_new_direction_after_turn,
    get_new_position_after_forward,
    get_valid_actions,
    is_collision,
    is_valid_action,
    move_writer_index,
    place_entities_on_grid,
    update_agent,
    update_shelf,
    write_to_observation,
)


@pytest.fixture
def fake_rware_env_state() -> State:
    """Create a fake rware environment state."""

    # create agents, shelves and grid
    def make_agent(
        x: chex.Array, y: chex.Array, direction: chex.Array, is_carrying: chex.Array
    ) -> Agent:
        return Agent(Position(x=x, y=y), direction=direction, is_carrying=is_carrying)

    def make_shelf(x: chex.Array, y: chex.Array, is_requested: chex.Array) -> Shelf:
        return Shelf(Position(x=x, y=y), is_requested=is_requested)

    # agent information
    xs = jnp.array([3, 1])
    ys = jnp.array([4, 7])
    dirs = jnp.array([2, 3])
    carries = jnp.array([0, 0])
    agents = jax.vmap(make_agent)(xs, ys, dirs, carries)

    # shelf information
    xs = jnp.array([1, 1, 1, 1, 2, 2, 2, 2])
    ys = jnp.array([1, 2, 7, 8, 1, 2, 7, 8])
    requested = jnp.array([0, 1, 1, 0, 0, 0, 1, 1])
    shelves = jax.vmap(make_shelf)(xs, ys, requested)

    # create grid
    grid = jnp.array(
        [
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 2, 0, 0, 0, 0, 3, 4, 0],
                [0, 5, 6, 0, 0, 0, 0, 7, 8, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ]
    )
    action_mask = jnp.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    state = State(
        grid=grid,
        agents=agents,
        shelves=shelves,
        request_queue=jnp.array([1, 2, 6, 7]),
        step_count=jnp.array([0], dtype=jnp.int32),
        action_mask=action_mask,
        key=jax.random.PRNGKey(42),
    )
    return state


def test_rware_utils__entity_placement(fake_rware_env_state: State) -> None:
    """Test entity placement on the warehouse grid floor."""
    state = fake_rware_env_state
    agents, shelves = state.agents, state.shelves
    empty_grid = jnp.zeros(state.grid.shape, dtype=jnp.int32)
    grid_with_agents_and_shelves = place_entities_on_grid(empty_grid, agents, shelves)

    # check that placement is the same as in fake grid
    assert jnp.all(grid_with_agents_and_shelves == state.grid)


def test_rware_utils__entity_update(fake_rware_env_state: State) -> None:
    """Test entity attribute (e.g. position, direction etc.) updating."""
    state = fake_rware_env_state
    agents = state.agents
    shelves = state.shelves

    # test updating agent position
    new_position = Position(x=2, y=4)
    agents_with_new_agent_0_position = update_agent(agents, 0, "position", new_position)
    agent_0 = tree_slice(agents_with_new_agent_0_position, 0)
    assert agent_0.position == new_position

    # test updating agent direction
    new_direction = 3
    agents_with_new_agent_0_direction = update_agent(
        agents, 0, "direction", new_direction
    )
    agent_0 = tree_slice(agents_with_new_agent_0_direction, 0)
    assert agent_0.direction == new_direction

    # test updating agent carrying
    new_is_carrying = 1
    agents_with_new_agent_0_carrying = update_agent(
        agents, 0, "is_carrying", new_is_carrying
    )
    agent_0 = tree_slice(agents_with_new_agent_0_carrying, 0)
    assert agent_0.is_carrying == new_is_carrying

    # test updating shelf position
    new_position = Position(x=1, y=3)
    shelves_with_new_shelf_0_position = update_shelf(
        shelves, 0, "position", new_position
    )
    shelf_0 = tree_slice(shelves_with_new_shelf_0_position, 0)
    assert shelf_0.position == new_position

    # test updating shelf requested
    new_is_requested = 1
    shelves_with_new_shelf_0_requested = update_shelf(
        shelves, 0, "is_requested", new_is_requested
    )
    shelf_0 = tree_slice(shelves_with_new_shelf_0_requested, 0)
    assert shelf_0.is_requested == new_is_requested


def test_rware_utils__get_new_direction(fake_rware_env_state: State) -> None:
    """Test the calculation of the new direction for an agent after turning."""
    state = fake_rware_env_state
    agents = state.agents
    agent = tree_slice(agents, 0)
    direction = agent.direction  # 2 (facing down)

    # turning: left, left, right, right, right
    actions = [2, 2, 3, 3, 3]
    expected_directions = [
        1,  # turn left -> facing right
        0,  # turn left -> facing up
        1,  # turn right -> facing right
        2,  # turn right -> face down
        3,  # turn right -> face left
    ]

    for action, expected_direction in zip(actions, expected_directions):
        new_direction = get_new_direction_after_turn(action, direction)
        assert new_direction == expected_direction
        direction = new_direction


def test_rware_utils__get_new_position(fake_rware_env_state: State) -> None:
    """Test the calculation of the new position for an agent after moving
    forward in a specific direction."""
    state = fake_rware_env_state
    grid = state.grid
    agents = state.agents
    agent = tree_slice(agents, 0)
    position = agent.position  # [x=3, y=4]
    directions = jnp.arange(4)

    # move forward once in each direction
    expected_positions = [
        Position(2, 4),  # facing up move forward
        Position(3, 5),  # facing right move forward
        Position(4, 4),  # facing down move forward
        Position(3, 3),  # facing left move forward
    ]

    for direction, expected_position in zip(directions, expected_positions):
        new_position = get_new_position_after_forward(grid, position, direction)
        assert new_position == expected_position


def test_rware_utils__is_collision(fake_rware_env_state: State) -> None:
    """Test the calculation of collisions between agents and other agents as well as
    agents carrying shelves and other shelves."""
    state = fake_rware_env_state
    grid = state.grid
    agents = state.agents

    # check no collision with original grid
    agent = tree_slice(agents, 0)
    action = 1  # forward
    collision = is_collision(grid, agent, action)
    assert bool(collision) is False

    # create grid with agents next to each other
    grid_prior_to_collision = jnp.array(
        [
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 2, 0, 0, 0, 0, 3, 4, 0],
                [0, 5, 6, 0, 0, 0, 0, 7, 8, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ]
    )

    # update agent zero to face agent 2
    agents = update_agent(agents, 0, "position", Position(1, 6))
    agents = update_agent(agents, 0, "direction", 1)
    agent = tree_slice(agents, 0)

    # check collision if moving forward
    collision = is_collision(grid_prior_to_collision, agent, action)
    assert bool(collision) is True


def test_rware_utils__is_valid_action(fake_rware_env_state: State) -> None:
    """Test the calculation of collisions between agents and other agents as well as
    agents carrying shelves and other shelves."""
    state = fake_rware_env_state
    grid = state.grid
    agents = state.agents

    # turn agent 2 around, and check no collision with shelf
    # i.e. agent is moving underneath shelf rack via highway
    agents = update_agent(agents, 1, "direction", 1)
    agent = tree_slice(agents, 1)
    action = 1  # forward
    action = is_valid_action(grid, agent, action)
    assert action == Action.FORWARD.value

    # Let agent 2 pick up shelf and move forward
    # to test collision with shelf when carrying
    # and convert to NOOP action
    agents = update_agent(agents, 1, "is_carrying", 1)
    agent = tree_slice(agents, 1)
    action = is_valid_action(grid, agent, action)
    assert action == Action.NOOP.value


def test_rware_utils__get_agent_view(fake_rware_env_state: State) -> None:
    """Test extracting the agent's view of other agents and shelves within
    its receptive field as set via a given sensor range."""
    state = fake_rware_env_state
    grid = jnp.array(
        [
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 2, 0, 0, 0, 0, 3, 4, 0],
                [0, 5, 6, 0, 0, 0, 0, 7, 8, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ]
    )
    agents = state.agents
    agents = update_agent(agents, 0, "position", Position(1, 6))
    agent = tree_slice(agents, 0)

    # get agent view with sensor range of 1
    sensor_range = 1
    agent_view_of_agents, agent_view_of_shelves = get_agent_view(
        grid, agent, sensor_range
    )

    # flattened agent view of other agents and shelves
    flat_agents = jnp.array([0, 0, 0, 0, 1, 2, 0, 0, 0])
    flat_shelves = jnp.array([0, 0, 0, 0, 0, 3, 0, 0, 7])

    assert jnp.array_equal(agent_view_of_agents, flat_agents)
    assert jnp.array_equal(agent_view_of_shelves, flat_shelves)

    # get agent view with sensor range of 2
    sensor_range = 2
    agent_view_of_agents, agent_view_of_shelves = get_agent_view(
        grid, agent, sensor_range
    )

    # flattened agent view of other agents and shelves
    flat_agents = jnp.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    flat_shelves = jnp.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 7, 8, 0, 0, 0, 0, 0]
    )

    assert jnp.array_equal(agent_view_of_agents, flat_agents)
    assert jnp.array_equal(agent_view_of_shelves, flat_shelves)


def test_rware_utils__calculate_num_observation_features() -> None:
    """Test the calculation of the size of the agent's observation
    vector based on sensor range."""
    sensor_range = 1
    num_obs_features = calculate_num_observation_features(sensor_range)
    assert num_obs_features == 66

    sensor_range = 2
    num_obs_features = calculate_num_observation_features(sensor_range)
    assert num_obs_features == 178


def test_rware_utils__observation_writer(fake_rware_env_state: State) -> None:
    """Test observation writer to write data to 1-d observation vector.
    Note that this test does not construct a full observation vector. It
    only tests basic functionality by writing the agent's view of itself
    and does not include writing agent view data from other agents/shelves."""
    state = fake_rware_env_state
    agents = state.agents
    agent = tree_slice(agents, 0)

    # write flattened observation just for the agent's own view
    obs = jnp.zeros(8, dtype=jnp.int32)
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

    # move index by one (keeping zero to indicate agent not on highway)
    idx = move_writer_index(idx, 1)

    assert jnp.array_equal(obs, jnp.array([3, 4, 0, 0, 0, 1, 0, 0]))
    assert idx == 8


def test_rware_utils__compute_action_mask(fake_rware_env_state: State) -> None:
    state = fake_rware_env_state
    grid = state.grid
    agents = state.agents

    action_mask = compute_action_mask(grid, agents)
    assert jnp.array_equal(action_mask[1], jnp.array([1, 1, 1, 1, 1]))

    # Let agent 2 turn around, pick up shelf and move forward
    # to test collision with shelf when carrying
    # which is an illegal action
    agents = update_agent(agents, 1, "direction", 1)
    agents = update_agent(agents, 1, "is_carrying", 1)

    action_mask = compute_action_mask(grid, agents)
    assert jnp.array_equal(action_mask[1], jnp.array([1, 0, 1, 1, 1]))


def test_rware_utils__get_valid_action(fake_rware_env_state: State) -> None:
    state = fake_rware_env_state
    grid = state.grid
    agents = state.agents
    actions = jnp.array([1, 1])  # forward

    action_mask = compute_action_mask(grid, agents)
    actions = get_valid_actions(actions, action_mask)
    jax.debug.print("action, {a}", a=actions)
    assert jnp.array_equal(actions, jnp.array([1, 1]))

    # Let agent 2 turn around, pick up shelf and move forward
    # to test collision with shelf when carrying
    # which is an illegal action
    agents = update_agent(agents, 1, "direction", 1)
    agents = update_agent(agents, 1, "is_carrying", 1)

    action_mask = compute_action_mask(grid, agents)
    actions = get_valid_actions(actions, action_mask)
    assert jnp.array_equal(actions, jnp.array([1, 0]))  # turn into noop action
