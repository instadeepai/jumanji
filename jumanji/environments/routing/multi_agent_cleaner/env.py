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

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.multi_agent_cleaner.specs import ObservationSpec
from jumanji.environments.routing.multi_agent_cleaner.types import State
from jumanji.types import Action, TimeStep


class Cleaner(Environment[State]):
    """A JAX implementation of the 'Multi-Agent Cleaner' game.

    - observation: Observation
        - grid: jax array (int) containing the state of the board:
            0 for dirty tile, 1 for clean tile, 2 for wall.
        - agents_locations: jax array (int) of size (num_agents, 2) containing
            the location of each agent on the board.
        - action_mask: jax array (bool) of size (num_agents, 4) stating for each agent
            if each of the four actions (up, right, down, left) is allowed.

    - action: jax array (int) of shape (num_agents,) containing the action for each agent.
        (0: up, 1: right, 2: down, 3: left)

    - reward: global reward, +1 every time a tile is cleaned.

    - episode termination:
        - All tiles are clean.
        - The number of steps is greater than the limit.
        - An invalid action is selected for any of the agents.

    - state: State
        - grid: jax array (int) containing the state of the board:
            0 for dirty tile, 1 for clean tile, 2 for a wall.
        - agents_locations: jax array (int) of size (num_agents, 2) containing
            the location of each agent on the board.
        - action_mask: jax array (bool) of size (num_agents, 4) stating for each agent
            if each of the four actions (up, right, down, left) is allowed.
        - step_count: the number of steps from the beginning of the environment.
        - key: jax random generation key. Ignored since the environment is deterministic.
    """

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        num_agents: int,
    ) -> None:
        """Instantiate an Cleaner environment.

        Args:
            grid_width: width of the grid.
            grid_height: height of the grid.
            num_agents: number of agents.
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid_shape = (self.grid_width, self.grid_height)
        self.num_agents = num_agents

    def __repr__(self) -> str:
        return (
            f"Cleaner(\n"
            f"\tgrid_width={self.grid_width!r},\n"
            f"\tgrid_height={self.grid_height!r},\n"
            f"\tnum_agents={self.num_agents!r}, \n"
            ")"
        )

    def observation_spec(self) -> ObservationSpec:
        """Specification of the observation of the Cleaner environment.

        Returns:
            ObservationSpec containing the specifications for all observation fields:
                - grid_spec: BoundedArray of int between 0 and 2 (inclusive),
                    same shape as the grid.
                - agent_locations_spec: BoundedArray of int, shape is (num_agents, 2).
                    Maximum value for the first column is grid_width,
                    and maximum value for the second is grid_height.
                - action_mask_spec: BoundedArray of bool, shape is (num_agent, 4).
        """
        grid_spec = specs.BoundedArray(self.grid_shape, int, 0, 2, "grid")
        agents_locations_spec = specs.BoundedArray(
            (self.num_agents, 2), int, [0, 0], self.grid_shape, "agents_locations"
        )
        action_mask_spec = specs.BoundedArray(
            (self.num_agents, 4), bool, False, True, "action_mask"
        )
        return ObservationSpec(
            grid_spec=grid_spec,
            agents_locations_spec=agents_locations_spec,
            action_mask_spec=action_mask_spec,
        )

    def action_spec(self) -> specs.BoundedArray:
        """Specification of the actions for the Cleaner environment.

        Returns:
            BoundedArray (int) between 0 and 3 (inclusive) of shape (num_agents,).
        """
        return specs.BoundedArray((self.num_agents,), int, 0, 3, "actions")

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:  # type: ignore
        pass  # type: ignore

    def step(self, state: State, action: Action) -> Tuple[State, TimeStep]:  # type: ignore
        pass  # type: ignore
