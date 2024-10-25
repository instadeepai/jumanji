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

import jax.numpy as jnp

from jumanji.environments.routing.lbf.env import LevelBasedForaging
from jumanji.environments.routing.lbf.observer import GridObserver, VectorObserver
from jumanji.environments.routing.lbf.types import Food, State

# create food and agents for grid that looks like:
# "AGENT" | EMPTY   | EMPTY   | EMPTY  | EMPTY | EMPTY
# EMPTY   | "AGENT" | EMPTY   | EMPTY  | EMPTY | EMPTY
# EMPTY   | "FOOD"  | "AGENT" | "FOOD" | EMPTY | EMPTY
# EMPTY   | EMPTY   | EMPTY   | EMPTY  | EMPTY | EMPTY
# EMPTY   | EMPTY   | "FOOD"  | EMPTY  | EMPTY | EMPTY
# EMPTY   | EMPTY   | EMPTY   | EMPTY  | EMPTY | EMPTY


# Test cases for VectorObserver class
def test_lbf_observer_initialization(lbf_env_2s: LevelBasedForaging) -> None:
    observer = VectorObserver(fov=2, grid_size=8, num_agents=2, num_food=2)
    assert observer.fov == lbf_env_2s.fov
    assert observer.grid_size == lbf_env_2s.grid_size
    assert observer.num_agents == lbf_env_2s.num_agents
    assert observer.num_food == lbf_env_2s.num_food


def test_vector_full_obs(state: State) -> None:
    observer = VectorObserver(fov=6, grid_size=6, num_agents=3, num_food=3)
    obs1 = observer.state_to_observation(state)
    expected_agent_0_view = jnp.array(
        [2, 1, 4, 2, 3, 4, 4, 2, 3, 0, 0, 1, 1, 1, 2, 2, 2, 4]
    )
    expected_agent_1_view = jnp.array(
        [2, 1, 4, 2, 3, 4, 4, 2, 3, 1, 1, 2, 0, 0, 1, 2, 2, 4]
    )
    expected_agent_2_view = jnp.array(
        [2, 1, 4, 2, 3, 4, 4, 2, 3, 2, 2, 4, 0, 0, 1, 1, 1, 2]
    )

    assert jnp.all(obs1.agents_view[0, :] == expected_agent_0_view)
    assert jnp.all(
        obs1.action_mask[0, :] == jnp.array([True, False, True, False, True, False])
    )
    assert jnp.all(obs1.agents_view[1, :] == expected_agent_1_view)
    assert jnp.all(
        obs1.action_mask[1, :] == jnp.array([True, True, False, True, True, True])
    )
    assert jnp.all(obs1.agents_view[2, :] == expected_agent_2_view)
    assert jnp.all(
        obs1.action_mask[2, :] == jnp.array([True, True, True, False, False, True])
    )

    # If agent1 and agent2 eat the food0
    eaten = jnp.array([True, False, False])
    food_items = Food(
        id=state.food_items.id,
        position=state.food_items.position,
        level=state.food_items.level,
        eaten=eaten,
    )
    state = state.replace(food_items=food_items)  # type: ignore

    obs2 = observer.state_to_observation(state)
    expected_agent_1_view = jnp.array(
        [-1, -1, 0, 2, 3, 4, 4, 2, 3, 1, 1, 2, 0, 0, 1, 2, 2, 4]
    )
    expected_agent_2_view = jnp.array(
        [-1, -1, 0, 2, 3, 4, 4, 2, 3, 2, 2, 4, 0, 0, 1, 1, 1, 2]
    )
    assert jnp.all(obs2.agents_view[1, :] == expected_agent_1_view)
    assert jnp.all(
        obs2.action_mask[1, :] == jnp.array([True, True, True, True, True, False])
    )
    assert jnp.all(obs2.agents_view[2, :] == expected_agent_2_view)
    assert jnp.all(
        obs2.action_mask[2, :] == jnp.array([True, True, True, True, False, True])
    )


def test_vector_partial_obs(state: State) -> None:
    observer = VectorObserver(fov=2, grid_size=6, num_agents=3, num_food=3)
    obs1 = observer.state_to_observation(state)
    expected_agent_0_view = jnp.array(
        [2, 1, 4, -1, -1, 0, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 4]
    )
    expected_agent_1_view = jnp.array(
        [2, 1, 4, 2, 3, 4, -1, -1, 0, 1, 1, 2, 0, 0, 1, 2, 2, 4]
    )
    expected_agent_2_view = jnp.array(
        [2, 1, 4, 2, 3, 4, 4, 2, 3, 2, 2, 4, 0, 0, 1, 1, 1, 2]
    )

    assert jnp.all(obs1.agents_view[0, :] == expected_agent_0_view)
    assert jnp.all(
        obs1.action_mask[0, :] == jnp.array([True, False, True, False, True, False])
    )
    assert jnp.all(obs1.agents_view[1, :] == expected_agent_1_view)
    assert jnp.all(
        obs1.action_mask[1, :] == jnp.array([True, True, False, True, True, True])
    )
    assert jnp.all(obs1.agents_view[2, :] == expected_agent_2_view)
    assert jnp.all(
        obs1.action_mask[2, :] == jnp.array([True, True, True, False, False, True])
    )

    # test eaten food is not visible
    eaten = jnp.array([True, False, False])
    food_items = Food(
        id=state.food_items.id,
        position=state.food_items.position,
        level=state.food_items.level,
        eaten=eaten,
    )
    state = state.replace(food_items=food_items)  # type: ignore

    obs2 = observer.state_to_observation(state)
    expected_agent_1_view = jnp.array(
        [-1, -1, 0, 2, 3, 4, -1, -1, 0, 1, 1, 2, 0, 0, 1, 2, 2, 4]
    )
    expected_agent_2_view = jnp.array(
        [-1, -1, 0, 2, 3, 4, 4, 2, 3, 2, 2, 4, 0, 0, 1, 1, 1, 2]
    )
    assert jnp.all(obs2.agents_view[1, :] == expected_agent_1_view)
    assert jnp.all(
        obs2.action_mask[1, :] == jnp.array([True, True, True, True, True, False])
    )
    assert jnp.all(obs2.agents_view[2, :] == expected_agent_2_view)
    assert jnp.all(
        obs2.action_mask[2, :] == jnp.array([True, True, True, True, False, True])
    )


def test_grid_observer(state: State) -> None:
    observer = GridObserver(fov=2, grid_size=6, num_agents=3, num_food=3)
    obs = observer.state_to_observation(state)

    expected_agent_0_view = jnp.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 2, 0],
                [0, 0, 0, 0, 4],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 4, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 1, 0, 1],
                [0, 0, 1, 0, 0],
            ],
        ]
    )

    expected_agent_1_view = jnp.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 2, 0, 0],
                [0, 0, 0, 4, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 4, 0, 4],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1],
                [0, 1, 0, 1, 1],
                [0, 1, 0, 0, 0],
                [0, 1, 1, 1, 1],
            ],
        ]
    )

    expected_agent_2_view = jnp.array(
        [
            [
                [1, 0, 0, 0, 0],
                [0, 2, 0, 0, 0],
                [0, 0, 4, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 4, 0, 4, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 3, 0, 0],
            ],
            [
                [0, 1, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 0, 1, 1],
            ],
        ]
    )

    assert jnp.all(obs.agents_view[0, :] == expected_agent_0_view)
    assert jnp.all(
        obs.action_mask[0, :] == jnp.array([True, False, True, False, True, False])
    )
    assert jnp.all(obs.agents_view[1, :] == expected_agent_1_view)
    assert jnp.all(
        obs.action_mask[1, :] == jnp.array([True, True, False, True, True, True])
    )
    assert jnp.all(obs.agents_view[2, :] == expected_agent_2_view)
    assert jnp.all(
        obs.action_mask[2, :] == jnp.array([True, True, True, False, False, True])
    )
