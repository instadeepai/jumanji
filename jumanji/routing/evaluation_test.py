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
import jax.random as random
import pytest

from jumanji.routing.env import Routing
from jumanji.routing.evaluation import (
    is_board_complete,
    is_episode_finished,
    proportion_connected,
    wire_length,
)
from jumanji.routing.types import State


class TestEvaluation:
    @pytest.fixture(scope="module")
    def env(self) -> Routing:
        """Creates the Routing environment."""
        return Routing(8, 8, 2)

    def test_evaluation__is_board_complete(self, env: Routing) -> None:
        """Tests evaluation method is_board_complete correctly returns True when agents have
        reached desired positions."""
        state, timestep = env.reset(random.PRNGKey(0))
        assert not is_board_complete(env, state.grid)
        assert not is_board_complete(env, timestep.observation[0])
        assert not is_board_complete(env, timestep.observation[1])

        grid = jnp.array([[3, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [6, 0, 7, 5]])
        state = State(
            key=random.PRNGKey(0),
            grid=grid,
            step=0,
            finished_agents=jnp.array([False, False]),
        )

        state, timestep = env.step(state, jnp.array([1, 1]))
        assert not is_board_complete(env, state.grid)
        state, timestep = env.step(state, jnp.array([0, 1]))
        assert is_board_complete(env, state.grid)

    def test_evaluation__proportion_connected(self, env: Routing) -> None:
        """Tests that proportion_connected returns the correct value when different numbers of
        agents are connected."""
        state, timestep = env.reset(random.PRNGKey(0))
        assert proportion_connected(env, state.grid) == 0.0

        grid = jnp.array([[3, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [6, 0, 7, 5]])
        state = State(
            key=random.PRNGKey(0),
            grid=grid,
            step=0,
            finished_agents=jnp.array([False, False]),
        )

        state, timestep = env.step(state, jnp.array([1, 1]))
        assert proportion_connected(env, state.grid) == 0.5
        state, timestep = env.step(state, jnp.array([0, 1]))
        assert proportion_connected(env, state.grid) == 1.0

    def test_evaluation__wire_length(self, env: Routing) -> None:
        """Tests that `wire_length` accurately counts the number of wires on the board."""
        grid = jnp.array([[3, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [6, 0, 7, 5]])
        state = State(
            key=random.PRNGKey(0),
            grid=grid,
            step=0,
            finished_agents=jnp.array([False, False]),
        )
        assert wire_length(env, state.grid) == 1
        state, timestep = env.step(state, jnp.array([1, 1]))
        assert wire_length(env, state.grid) == 3
        state, timestep = env.step(state, jnp.array([0, 1]))
        assert wire_length(env, state.grid) == 4

    def test_evaluation__is_episode_finished(self, env: Routing) -> None:
        """Tests evaluation method is_board_complete correctly returns True when agents have
        reached desired positions."""
        state, timestep = env.reset(random.PRNGKey(0))
        assert not is_episode_finished(env, state.grid)
        assert not is_episode_finished(env, timestep.observation[0])
        assert not is_episode_finished(env, timestep.observation[1])

        grid = jnp.array([[3, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [6, 0, 7, 5]])
        state = State(
            key=random.PRNGKey(0),
            grid=grid,
            step=0,
            finished_agents=jnp.array([False, False]),
        )

        state, timestep = env.step(state, jnp.array([1, 1]))
        assert not is_episode_finished(env, state.grid)
        state, timestep = env.step(state, jnp.array([0, 1]))
        assert is_episode_finished(env, state.grid)

        state = State(
            key=random.PRNGKey(0),
            grid=grid,
            step=0,
            finished_agents=jnp.array([False, False]),
        )

        state, timestep = env.step(state, jnp.array([1, 1]))
        assert not is_episode_finished(env, state.grid)
        state, timestep = env.step(state, jnp.array([0, 2]))
        state, timestep = env.step(state, jnp.array([0, 2]))
        state, timestep = env.step(state, jnp.array([0, 3]))
        state, timestep = env.step(state, jnp.array([0, 3]))
        state, timestep = env.step(state, jnp.array([0, 4]))
        state, timestep = env.step(state, jnp.array([0, 1]))
        assert is_episode_finished(env, state.grid)
