import jax
import jumanji
import numpy as np
import time
from ic_routing_board_generation.ic_routing.env import Routing
from ic_routing_board_generation.interface.board_generator_interface import \
    BoardName
from jumanji.environments.combinatorial.routing.evaluation import \
    is_board_complete, wire_length, proportion_connected
from jumanji.types import StepType


class Route:
    def __init__(self, board_init: BoardName = BoardName.BFS_BASE, **kwargs):
        """Class attribute instantation.

        Args:
            board_init (optional): One of 'random' or 'randy'. An optional keyword that is used 
            to make code work in case no other **kwargs are provided.
        """
        self.board_init = board_init
        self.__dict__.update(kwargs)
        self.reinitialisation_counter = 0

    def bootstrap(self):
        # TODO (DK): You can now gate this using the enum (use .value from an enum)
        if self.board_init == 'random':
            return "random pins"
        elif self.board_init == 'randy':
            return "solvable (randy_v1 pins)"
        else:
            return "other, custom"

    def insantiate_random_board(self, **kwargs):
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
        return env

    def insantiate_board(self, **kwargs):
        if 'instance_generator_type' in kwargs:
            self.board_init = kwargs['instance_generator_type']

        if self.board_init == 'random':
            env = self.insantiate_random_board(**kwargs)

        else:
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

    def route_for_benchmarking(self, number_of_boards: int = 100, **kwargs):
        env, key, state, timestep = self.insantiate_board(**kwargs)
        env = jumanji.wrappers.AutoResetWrapper(env)
        total_reward = 0
        state_grid = None
        step_counter = 0
        board_counter = 0

        total_rewards = []
        was_board_filled = []
        total_wire_lengths = []
        proportion_wires_connected = []
        number_of_steps = []
        # TODO (MW): replace step counter with state.step - related to bug on timestep.LAST
        while board_counter < number_of_boards:
            action = self.act(env)
            state, timestep = self.step(env, state, action)

            # TODO (Marta / Danila): The timestep never goes to timestep.LAST which causes a bug
            if timestep.step_type == StepType.FIRST:
                was_board_filled.append(is_board_complete(env, state_grid))
                total_wire_lengths.append(int(wire_length(env, state_grid)))
                total_rewards.append(total_reward)
                number_of_steps.append(step_counter)
                proportion_wires_connected.append(
                    proportion_connected(env, state_grid))
                total_reward = 0
                step_counter = 0
                board_counter += 1
                self.reinitialisation_counter += 1
            else:
                # workaround to keep track of grid before it resets
                state_grid = state.grid  # TODO (MW): remove this workaround after timestep.LAST bug is fixed
                total_reward += timestep.reward
                state_grid = state.grid
                step_counter += 1

        time.sleep(1)
        env.close()
        return total_rewards, was_board_filled, total_wire_lengths, proportion_wires_connected, number_of_steps


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
