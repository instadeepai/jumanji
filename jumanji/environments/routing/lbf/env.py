from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp

import jumanji.environments.routing.lbf.utils as utils
from jumanji.env import Environment
from jumanji.environments.routing.lbf.generator import UniformRandomGenerator
from jumanji.environments.routing.lbf.types import State
from jumanji.types import TimeStep


class LevelBasedForaging(Environment[State]):
    def __init__(self, generator: Optional[UniformRandomGenerator]) -> None:
        super().__init__()

        self._generator = generator or UniformRandomGenerator(
            grid_size=10, num_agents=4, num_food=5, max_agent_level=3, max_food_level=3
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        state = self._generator(key)
        return state, self._state_to_timestep(state)

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        moved_agents = jax.vmap(utils.move, in_axes=(0, None, 0))(
            state.agents, state.grid, action
        )
        # todo: make sure that agent doesn't move into food
        # check that no two agent share the same position
        moved_agents = utils.fix_collisions(moved_agents, state.agents)
        # todo: might need to vmap
        grid = utils.place_agent_on_grid(moved_agents, state.grid)
        # eat food
        foods, grid = utils.eat(moved_agents, state.foods, grid)

        # todo: reward

        state = State(
            agents=moved_agents,
            foods=foods,
            grid=grid,
            step_count=state.step_count + 1,
            key=state.key,
        )

        return state, self._state_to_timestep(state)

    def _state_to_timestep(self, state: State) -> TimeStep:
        pass
