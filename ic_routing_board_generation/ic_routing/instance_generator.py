import abc

import numpy as np

from chex import Array, PRNGKey
from typing import Tuple, Optional
import jax
import jax.numpy as jnp
from jax import random

from ic_routing_board_generation.interface.board_generator_interface import \
    BoardName, BoardGenerator
from jumanji.environments.combinatorial.routing import State


class InstanceGenerator(abc.ABC):
    """Defines the abstract `InstanceGenerator` base class. An `InstanceGenerator` is responsible
    for generating an instance when the environment is reset.
    """
    def __init__(
        self, rows: int = 4, cols: int = 4, num_agents: int = 3,
        board_generator: BoardName = BoardName.BFS_BASE,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.num_agents = num_agents
        self.board_generator = board_generator

class UniversalInstanceGenerator(InstanceGenerator):
    """Instance generator using a custom board (none of randy or random)."""
    def __init__(
        self, rows: int, cols: int, num_agents: int,
        board_generator: BoardName,
    ) -> None:
        super().__init__(rows, cols, num_agents, board_generator)
    def __call__(self, key:PRNGKey) -> Tuple[Array, State]:
        board_class = BoardGenerator.get_board_generator(board_enum=self.board_generator)
        board = board_class(self.rows, self.cols, self.num_agents)
        pins = board.return_training_board() # edit this line here
        grid = jnp.array(pins, int)

        state = State(
            key=key,
            grid=grid,
            step=jnp.array(0, int),
            finished_agents=jnp.zeros(self.num_agents, bool)
        )
        return grid, state

class NewInstanceGenerator(abc.ABC):
    def __init__(
        self, rows:int=4, cols:int=4, num_agents:int=3, board_enum: BoardName = BoardName.BFS_BASE,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.num_agents = num_agents
        self.board_enum = board_enum


class MartasNewInstanceGenerator(NewInstanceGenerator):
    def __init__(
        self, rows: int=4, cols: int=4, num_agents: int=3, board_enum: BoardName=BoardName.BFS_BASE,
        ) -> None:
        super().__init__(rows, cols, num_agents, board_enum)

    def __call__(self, key:PRNGKey) -> Tuple[Array, State]:
        """Call method responsible for generating a new state. It returns a board generated
        according to randy's v1 board generator.

        Args:
            key : NOT NECESSARY; here used so that other parts of the code don't crash.

        Returns:
            A Routing (Randyv1 ~15/01/2023) State.
        """
        board_class = BoardGenerator.get_board_generator(board_enum=self.board_enum)
        board_class_instance = board_class(rows=self.rows, cols=self.cols, num_agents=self.num_agents)
        pins = board_class_instance.return_training_board()
        # pins, _, _ = randy_route.board_generator(x_dim=self.cols, y_dim=self.rows, target_wires=self.num_agents)
        grid = jnp.array(pins, int)

        state = State(
            key=key,
            grid=grid,
            step=jnp.array(0, int),
            finished_agents=jnp.zeros(self.num_agents, bool)
        )
        return grid, state

class RandomInstanceGenerator(InstanceGenerator):
    """Instance generator that generates random pin locations. This generation works as follows:
    An empty board is first initialised with all zeros. `num_agents` PRNGKeys are generated of size
    2. For each key, two randomly chosen location on the board are generated, corresponding to HEAD 
    and TARGET for that agent. This is repeated `num_agents` times.

    Example: # TODO!!!
        ```python
        env = BinPack(instance_generator_type="random")
        key = jax.random.key(0)
        reset_state = env.instance_generator(key)
        env.render(reset_state)
        solution = env.instance_generator.generate_solution(key)
        env.render(solution)
        ```"""
    def __init__(
        self, rows: int=4, cols: int=4, num_agents: int=3
        ) -> None:
        super().__init__(rows, cols, num_agents)

    def __call__(self, key:PRNGKey) -> Tuple[Array, State]:
        """Call method responsible for generating a new state.

        Args:
            key: jax random key for generating `num_agents` jax random keys.

        Returns:
            A Routing State that corresponds to a random board initialisation.
        """
        grid = jnp.zeros((self.rows, self.cols), int)
        state_key, spawn_key = random.split(key)
        spawn_keys = random.split(spawn_key, self.num_agents)

        def spawn_scan(grid_and_agent_id: Tuple[Array, int], key: PRNGKey) -> Tuple[Tuple[Array, int], None]:
            
            grid, agent_id = grid_and_agent_id
            grid = self._spawn_agent(grid, key, agent_id)
            return (grid, agent_id + 1), None

        (grid, _), _ = jax.lax.scan(spawn_scan, (grid, 0), spawn_keys)
        state = State(
            key=state_key,
            grid=grid,
            step=jnp.array(0, int),
            finished_agents=jnp.zeros(self.num_agents, bool),
        )
        return grid, state
