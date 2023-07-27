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
    def __init__(self, generator: Optional[UniformRandomGenerator], fov: int) -> None:
        super().__init__()

        self._generator = generator or UniformRandomGenerator(
            grid_size=10, num_agents=4, num_food=5, max_agent_level=3, max_food_level=3
        )
        self.fov = fov

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        state = self._generator(key)
        return state, self._state_to_timestep(state)

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        moved_agents = jax.vmap(utils.move, (0, 0, None, None, None))(
            state.agents,
            action,
            state.foods,
            self._generator.grid_size,
            self._generator.grid_size,  # todo: just take in grid_size
        )
        # todo: make sure that agent doesn't move into food
        # check that no two agent share the same position
        moved_agents = utils.fix_collisions(moved_agents, state.agents)
        # eat food
        eaten, foods = jax.vmap(utils.eat, (None, 0))(moved_agents, state.foods)

        grid = jnp.zeros((self._generator.grid_size, self._generator.grid_size))
        # todo: might need to vmap
        grid = jax.vmap(utils.place_agent_on_grid)(moved_agents, gird)

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

    def _get_agent_view(self, grid: chex.Array, agent: Agent) -> chex.Array:
        pass
