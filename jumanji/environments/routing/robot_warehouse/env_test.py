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

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from jax import random

from jumanji.environments.routing.robot_warehouse.env import RobotWarehouse
from jumanji.environments.routing.robot_warehouse.types import State
from jumanji.testing.env_not_smoke import (
    check_env_does_not_smoke,
    check_env_specs_does_not_smoke,
)
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.tree_utils import tree_slice
from jumanji.types import TimeStep


def test_robot_warehouse__specs(robot_warehouse_env: RobotWarehouse) -> None:
    """Validate environment specs conform to the expected shapes and values"""
    action_spec = robot_warehouse_env.action_spec
    observation_spec = robot_warehouse_env.observation_spec

    assert observation_spec.agents_view.shape == (2, 66)  # type: ignore
    assert action_spec.num_values.shape[0] == robot_warehouse_env.num_agents
    assert action_spec.num_values[0] == 5


def test_robot_warehouse__reset(robot_warehouse_env: RobotWarehouse) -> None:
    """Validate the jitted reset of the environment."""
    chex.clear_trace_counter()
    reset_fn = jax.jit(chex.assert_max_traces(robot_warehouse_env.reset, n=1))

    key1, key2 = random.PRNGKey(0), random.PRNGKey(1)
    state1, timestep1 = reset_fn(key1)
    state2, timestep2 = reset_fn(key2)

    assert isinstance(timestep1, TimeStep)
    assert isinstance(state1, State)
    assert state1.step_count == 0
    assert state1.grid.shape == (2, *robot_warehouse_env.grid_size)
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # reset function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(state1)
    # Check random initialization
    assert not jnp.all(state1.key == state2.key)
    assert not jnp.all(state1.grid == state2.grid)
    assert state1.step_count == state2.step_count


def test_robot_warehouse__agent_observation(
    deterministic_robot_warehouse_env: Tuple[RobotWarehouse, State, TimeStep],
) -> None:
    """Validate the agent observation function."""
    env, state, timestep = deterministic_robot_warehouse_env
    state, timestep = env.step(state, jnp.array([0, 0]))

    # agent 1 obs
    agent1_own_view = jnp.array([3, 4, 0, 0, 0, 1, 0, 1])
    agent1_other_agents_view = jnp.array(8 * [0, 0, 0, 0, 0])
    agent1_shelf_view = jnp.array(9 * [0, 0])
    agent1_obs = jnp.hstack(
        [agent1_own_view, agent1_other_agents_view, agent1_shelf_view]
    )

    # agent 2 obs
    agent2_own_view = jnp.array([1, 7, 0, 0, 0, 0, 1, 0])
    agent2_other_agents_view = jnp.array(8 * [0, 0, 0, 0, 0])
    agent2_shelf_view = jnp.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1]
    )
    agent2_obs = jnp.hstack(
        [agent2_own_view, agent2_other_agents_view, agent2_shelf_view]
    )

    assert jnp.all(timestep.observation.agents_view[0] == agent1_obs)
    assert jnp.all(timestep.observation.agents_view[1] == agent2_obs)


def test_robot_warehouse__step(robot_warehouse_env: RobotWarehouse) -> None:
    """Validate the jitted step function of the environment."""
    chex.clear_trace_counter()

    step_fn = chex.assert_max_traces(robot_warehouse_env.step, n=1)
    step_fn = jax.jit(step_fn)

    state_key, action_key1, action_key2 = random.split(random.PRNGKey(10), 3)
    state, timestep = robot_warehouse_env.reset(state_key)

    # Sample two different actions
    action1, action2 = random.choice(
        key=action_key1,
        a=jnp.arange(5),
        shape=(2,),
        replace=False,
    )

    action1 = jnp.zeros((robot_warehouse_env.num_agents,), int).at[0].set(action1)
    action2 = jnp.zeros((robot_warehouse_env.num_agents,), int).at[0].set(action2)

    new_state1, timestep1 = step_fn(state, action1)

    # Check that rewards have the correct number of dimensions
    assert jnp.ndim(timestep1.reward) == 0
    assert jnp.ndim(timestep.reward) == 0
    # Check that discounts have the correct number of dimensions
    assert jnp.ndim(timestep1.discount) == 0
    assert jnp.ndim(timestep.discount) == 0
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # step function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(new_state1)
    # Check that the state has changed
    assert new_state1.step_count != state.step_count
    assert not jnp.all(new_state1.grid != state.grid)
    # Check that two different actions lead to two different states
    new_state2, timestep2 = step_fn(state, action2)
    assert not jnp.all(new_state1.grid != new_state2.grid)

    # Check that the state update and timestep creation work as expected
    agents = state.agents
    agent = tree_slice(agents, 1)
    x = agent.position.x
    y = agent.position.y

    # turning and moving actions
    actions = [2, 2, 3, 3, 1, 3, 1]

    # Note: starting direction is 3 (facing left)
    new_locs = [
        (x, y, 2),  # turn left -> facing down
        (x, y, 1),  # turn left -> facing right
        (x, y, 2),  # turn right -> facing down
        (x, y, 3),  # turn right -> face left
        (x, y - 1, 3),  # move forward -> move left
        (x, y - 1, 0),  # turn right -> face up
        (x - 1, y - 1, 0),  # move forward -> move up
    ]

    for action, new_loc in zip(actions, new_locs):
        state, timestep = step_fn(state, jnp.array([action, action]))
        agent1_info = tree_slice(state.agents, 1)
        agent1_loc = (
            agent1_info.position.x,
            agent1_info.position.y,
            agent1_info.direction,
        )
        assert agent1_loc == new_loc


def test_robot_warehouse__does_not_smoke(robot_warehouse_env: RobotWarehouse) -> None:
    """Validate that we can run an episode without any errors."""
    check_env_does_not_smoke(robot_warehouse_env)


def test_robot_warehouse__specs_does_not_smoke(
    robot_warehouse_env: RobotWarehouse,
) -> None:
    """Test that we can access specs without any errors."""
    check_env_specs_does_not_smoke(robot_warehouse_env)


def test_robot_warehouse__time_limit(robot_warehouse_env: RobotWarehouse) -> None:
    """Validate the terminal reward."""
    step_fn = jax.jit(robot_warehouse_env.step)
    state_key = random.PRNGKey(10)
    state, timestep = robot_warehouse_env.reset(state_key)
    assert timestep.first()

    for _ in range(robot_warehouse_env.time_limit - 1):
        state, timestep = step_fn(state, jnp.array([0, 0]))

    assert timestep.mid()
    state, timestep = step_fn(state, jnp.array([0, 0]))
    assert timestep.last()


def test_robot_warehouse__truncation(
    deterministic_robot_warehouse_env: Tuple[RobotWarehouse, State, TimeStep],
) -> None:
    """Validate episode truncation based on set time limit."""
    robot_warehouse_env, state, timestep = deterministic_robot_warehouse_env
    step_fn = jax.jit(robot_warehouse_env.step)

    # truncation
    for _ in range(robot_warehouse_env.time_limit):
        state, timestep = step_fn(state, jnp.array([0, 0]))

    assert timestep.last()
    # note the line below should be used to test for truncation
    # but since we instead use termination inside the env code
    # for training capatibility, we check for omit this check
    # assert not jnp.all(timestep.discount == 0)


def test_robot_warehouse__truncate_upon_collision(
    deterministic_robot_warehouse_env: Tuple[RobotWarehouse, State, TimeStep],
) -> None:
    """Validate episode terminates upon collision of agents."""
    robot_warehouse_env, state, timestep = deterministic_robot_warehouse_env
    step_fn = jax.jit(robot_warehouse_env.step)

    # actions for agent 1 to collide with agent 2
    actions = [3, 3, 1, 1, 3, 1, 1, 1]

    # take actions until collision
    for action in actions:
        state, timestep = step_fn(state, jnp.array([action, 0]))

    assert timestep.last()
    # TODO: uncomment once we have changed termination
    # in the env code to truncation (also see above)
    # assert not jnp.all(timestep.discount == 0)


def test_robot_warehouse__reward_in_goal(
    deterministic_robot_warehouse_env: Tuple[RobotWarehouse, State, TimeStep],
) -> None:
    """Validate goal reward behavior."""
    robot_warehouse_env, state, timestep = deterministic_robot_warehouse_env
    step_fn = jax.jit(robot_warehouse_env.step)

    # actions for agent 1 to deliver shelf to goal
    actions = [4, 1, 2, 1, 1, 1, 3]

    # check no reward is given when not at goal state
    for action in actions:
        state, timestep = step_fn(state, jnp.array([0, action]))
        assert timestep.reward == 0

    # final step to delivery shelf
    state, timestep = step_fn(state, jnp.array([0, 1]))
    assert timestep.reward == 1
