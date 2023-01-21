import jax
import jumanji
import numpy as np
import time
from routing.env_test import Routing
# from typing import Any, Optional
# from routing import env_viewer



class Route:
    def __init__(self, board_init:str='random', **kwargs):
        """Class attribute instantation.

        Args:
            board_init (optional): One of 'random' or 'randy'. An optional keyword that is used 
            to make code work in case no other **kwargs are provided.
        """
        self.board_init = board_init
        self.__dict__.update(kwargs)

    def bootstrap(self):
        if self.board_init == 'random':
            return "random pins"
        else:
            return "solvable (randy_v1 pins)"

    def insantiate_board(self, **kwargs):
        if 'instance_generator_type' in kwargs:
            self.board_init = kwargs['instance_generator_type']
        
        if self.board_init == 'random':
            if 'rows' in kwargs:
                rows = kwargs['rows']
            if 'cols' in kwargs:
                cols = kwargs['cols']
            if 'rows' not in kwargs and 'cols' not in kwargs:
                print('No `rows` or `cols` provided as **kwarg initialisation, instantiating the usual 8x8.')
                rows, cols = 8, 8

            if rows not in [8, 12, 16] and cols not in [8, 12, 16]:
                print('Rows/cols should be one of 8, 12, or 16.')

            if rows == 8 or cols == 8:
                print('Instantiating the 8x8 Routing Board...')
                env = jumanji.make('Routing-n3-8x8-v0')
            elif rows == 12 or cols == 12:
                print('Instantiating a 12x12 Routing Board...')
                env = jumanji.make('Routing-n4-12x12-v0')
            elif rows == 16 or cols == 18:
                print('Instantiating a 16x16 Routing Board...')
                env = jumanji.make('Routing-n5-16x16-v0')

        elif self.board_init == 'randy':
            env = Routing(**kwargs)
            print(f"Instantiating the {env.rows}x{env.cols} Routing Board for {env.num_agents} agents...")

        key = jax.random.PRNGKey(0)
        state, timestep = jax.jit(env.reset)(key)
        return env, key, state, timestep

    def act(self, env, key=None):
        action = np.random.randint(low=0, high=5, size=env.num_agents, dtype='int32')
        return action

    def step(self, env, state, action):
        state, timestep = jax.jit(env.step)(state, action)
        return state, timestep

    def route(self, time_steps:int=100, fps:int=30, **kwargs):
        env, key, state, timestep = self.insantiate_board(**kwargs)
        print(f'Routing a {self.bootstrap()} board over {time_steps} time steps at {fps} fps.')
        env = jumanji.wrappers.AutoResetWrapper(env)
        
        for _ in range(time_steps):
            time.sleep(1/fps)
            env.render(state)
            action = self.act(env)
            state, timestep = self.step(env, state, action)

        time.sleep(1)
        env.close() # I am having issues here figuring out how to close pygame correctly.
        print(f'Routed {time_steps} time steps.')
pass


# a prior version of route subclassed:
"""
# class custom_Routing(Routing):
#     def __init__(self, 
#                  rows: int = 4, 
#                  cols: int = 4, 
#                  num_agents: int = 3, 
#                  reward_per_timestep: float = -0.03, reward_for_connection: float = 0.1, 
#                  reward_for_blocked: float = -0.1, reward_for_noop: float = -0.01, 
#                  step_limit: int = 150, 
#                  reward_for_terminal_step: float = -0.1, 
#                  renderer: Optional[env_viewer.RoutingViewer] = None):
        
#         super().__init__(rows, cols, num_agents, reward_per_timestep, 
#                          reward_for_connection, reward_for_blocked, 
#                          reward_for_noop, step_limit, 
#                          reward_for_terminal_step, renderer)

#     def reset(self, key):
#         pins, _, _ = rr.board_generator(x_dim=4, y_dim=4, target_wires=self.num_agents)
#         grid = jnp.array(pins, int)
#         observations = jax.vmap(functools.partial(self._agent_observation, grid))(jnp.arange(self.num_agents, dtype=int))
#         timestep = restart(observation=observations, shape=self.num_agents)
        
#         state = State(
#             key=key,
#             grid=grid,
#             step=jnp.array(0, int),
#             finished_agents=jnp.zeros(self.num_agents, bool),
#         )
#         return state, timestep
"""