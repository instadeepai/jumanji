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

from jumanji.environments.routing.connector.generator.base import Generator
from jumanji.environments.routing.connector.types import Agent, State
from jumanji.environments.routing.connector.utils import get_action_masks, get_position, get_target


class UniformRandomGenerator(Generator):
    """Randomly generates `Connector` grids that may or may not be solvable. This generator places
    start and target positions uniformly at random on the grid.
    """

    def __init__(self, grid_size: int, num_agents: int) -> None:
        """Instantiates a `UniformRandomGenerator`.

        Args:
            grid_size: size of the square grid to generate.
            num_agents: number of agents/paths on the grid.
        """
        super().__init__(grid_size, num_agents)

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `Connector` state that contains the grid and the agents' layout.

        Returns:
            A `Connector` state.
        """
        key, pos_key = jax.random.split(key)
        starts_flat, targets_flat = jax.random.choice(
            key=pos_key,
            a=jnp.arange(self.grid_size**2),
            shape=(2, self.num_agents),  # Start and target positions for all agents
            replace=False,  # Start and target positions cannot overlap
        )

        # Create 2D points from the flat arrays.
        starts = jnp.divmod(starts_flat, self.grid_size)
        targets = jnp.divmod(targets_flat, self.grid_size)

        # Get the agent values for starts and positions.
        agent_position_values = jax.vmap(get_position)(jnp.arange(self.num_agents))
        agent_target_values = jax.vmap(get_target)(jnp.arange(self.num_agents))

        # Create empty grid.
        grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)

        # Place the agent values at starts and targets.
        grid = grid.at[starts].set(agent_position_values)
        grid = grid.at[targets].set(agent_target_values)

        # Create the agent pytree that corresponds to the grid.
        agents = jax.vmap(Agent)(
            id=jnp.arange(self.num_agents),
            start=jnp.stack(starts, axis=1),
            target=jnp.stack(targets, axis=1),
            position=jnp.stack(starts, axis=1),
        )

        step_count = jnp.array(0, jnp.int32)
        action_mask = get_action_masks(agents, grid)

        return State(
            key=key, grid=grid, step_count=step_count, agents=agents, action_mask=action_mask
        )
