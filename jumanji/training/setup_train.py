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
import optax
from omegaconf import DictConfig

import jumanji
from jumanji.env import Environment
from jumanji.environments import (
    CVRP,
    MMST,
    TSP,
    BinPack,
    Cleaner,
    Connector,
    Game2048,
    GraphColoring,
    JobShop,
    Knapsack,
    Maze,
    Minesweeper,
    MultiCVRP,
    RobotWarehouse,
    RubiksCube,
    Snake,
    Sudoku,
    Tetris,
)
from jumanji.training import networks
from jumanji.training.agents.a2c import A2CAgent
from jumanji.training.agents.base import Agent
from jumanji.training.agents.random import RandomAgent
from jumanji.training.evaluator import Evaluator
from jumanji.training.loggers import (
    Logger,
    NeptuneLogger,
    NoOpLogger,
    TensorboardLogger,
    TerminalLogger,
)
from jumanji.training.networks.actor_critic import ActorCriticNetworks
from jumanji.training.networks.protocols import RandomPolicy
from jumanji.training.types import ActingState, TrainingState
from jumanji.wrappers import VmapAutoResetWrapper


def setup_logger(cfg: DictConfig) -> Logger:
    logger: Logger
    # Log only once if there are multiple hosts on the pod.
    if jax.process_index() != 0:
        return NoOpLogger()
    if cfg.logger.type == "tensorboard":
        logger = TensorboardLogger(
            name=cfg.logger.name, save_checkpoint=cfg.logger.save_checkpoint
        )
    elif cfg.logger.type == "neptune":
        logger = NeptuneLogger(
            name=cfg.logger.name,
            project="InstaDeep/jumanji",
            cfg=cfg,
            save_checkpoint=cfg.logger.save_checkpoint,
        )
    elif cfg.logger.type == "terminal":
        logger = TerminalLogger(
            name=cfg.logger.name, save_checkpoint=cfg.logger.save_checkpoint
        )
    else:
        raise ValueError(
            f"logger expected in ['neptune', 'tensorboard', 'terminal'], got {cfg.logger}."
        )
    return logger


def _make_raw_env(cfg: DictConfig) -> Environment:
    return jumanji.make(cfg.env.registered_version)


def setup_env(cfg: DictConfig) -> Environment:
    env = _make_raw_env(cfg)
    env = VmapAutoResetWrapper(env)
    return env


def setup_agent(cfg: DictConfig, env: Environment) -> Agent:
    agent: Agent
    if cfg.agent == "random":
        random_policy = _setup_random_policy(cfg, env)
        agent = RandomAgent(
            env=env,
            n_steps=cfg.env.training.n_steps,
            total_batch_size=cfg.env.training.total_batch_size,
            random_policy=random_policy,
        )
    elif cfg.agent == "a2c":
        actor_critic_networks = _setup_actor_critic_neworks(cfg, env)
        optimizer = optax.adam(cfg.env.a2c.learning_rate)
        agent = A2CAgent(
            env=env,
            n_steps=cfg.env.training.n_steps,
            total_batch_size=cfg.env.training.total_batch_size,
            actor_critic_networks=actor_critic_networks,
            optimizer=optimizer,
            normalize_advantage=cfg.env.a2c.normalize_advantage,
            discount_factor=cfg.env.a2c.discount_factor,
            bootstrapping_factor=cfg.env.a2c.bootstrapping_factor,
            l_pg=cfg.env.a2c.l_pg,
            l_td=cfg.env.a2c.l_td,
            l_en=cfg.env.a2c.l_en,
        )
    else:
        raise ValueError(
            f"Expected agent name to be in ['random', 'a2c'], got {cfg.agent}."
        )
    return agent


def _setup_random_policy(  # noqa: CCR001
    cfg: DictConfig, env: Environment
) -> RandomPolicy:
    assert cfg.agent == "random"
    if cfg.env.name == "bin_pack":
        assert isinstance(env.unwrapped, BinPack)
        random_policy = networks.make_random_policy_bin_pack(bin_pack=env.unwrapped)
    elif cfg.env.name == "snake":
        assert isinstance(env.unwrapped, Snake)
        random_policy = networks.make_random_policy_snake()
    elif cfg.env.name == "tsp":
        assert isinstance(env.unwrapped, TSP)
        random_policy = networks.make_random_policy_tsp()
    elif cfg.env.name == "knapsack":
        assert isinstance(env.unwrapped, Knapsack)
        random_policy = networks.make_random_policy_knapsack()
    elif cfg.env.name == "job_shop":
        assert isinstance(env.unwrapped, JobShop)
        random_policy = networks.make_random_policy_job_shop()
    elif cfg.env.name == "cvrp":
        assert isinstance(env.unwrapped, CVRP)
        random_policy = networks.make_random_policy_cvrp()
    elif cfg.env.name == "multi_cvrp":
        assert isinstance(env.unwrapped, MultiCVRP)
        random_policy = networks.make_random_policy_multicvrp()
    elif cfg.env.name == "rubiks_cube":
        assert isinstance(env.unwrapped, RubiksCube)
        random_policy = networks.make_random_policy_rubiks_cube(
            rubiks_cube=env.unwrapped
        )
    elif cfg.env.name == "minesweeper":
        assert isinstance(env.unwrapped, Minesweeper)
        random_policy = networks.make_random_policy_minesweeper(
            minesweeper=env.unwrapped
        )
    elif cfg.env.name == "game_2048":
        assert isinstance(env.unwrapped, Game2048)
        random_policy = networks.make_random_policy_game_2048()
    elif cfg.env.name == "sudoku":
        assert isinstance(env.unwrapped, Sudoku)
        random_policy = networks.make_random_policy_sudoku(sudoku=env.unwrapped)
    elif cfg.env.name == "cleaner":
        assert isinstance(env.unwrapped, Cleaner)
        random_policy = networks.make_random_policy_cleaner()
    elif cfg.env.name == "maze":
        assert isinstance(env.unwrapped, Maze)
        random_policy = networks.make_random_policy_maze()
    elif cfg.env.name == "connector":
        assert isinstance(env.unwrapped, Connector)
        random_policy = networks.make_random_policy_connector()
    elif cfg.env.name == "tetris":
        assert isinstance(env.unwrapped, Tetris)
        random_policy = networks.make_random_policy_tetris(tetris=env.unwrapped)
    elif cfg.env.name == "mmst":
        assert isinstance(env.unwrapped, MMST)
        random_policy = networks.make_random_policy_mmst()
    elif cfg.env.name == "robot_warehouse":
        assert isinstance(env.unwrapped, RobotWarehouse)
        random_policy = networks.make_random_policy_robot_warehouse()
    elif cfg.env.name == "graph_coloring":
        assert isinstance(env.unwrapped, GraphColoring)
        random_policy = networks.make_random_policy_graph_coloring()
    else:
        raise ValueError(f"Environment name not found. Got {cfg.env.name}.")
    return random_policy


def _setup_actor_critic_neworks(  # noqa: CCR001
    cfg: DictConfig, env: Environment
) -> ActorCriticNetworks:
    assert cfg.agent == "a2c"
    if cfg.env.name == "bin_pack":
        assert isinstance(env.unwrapped, BinPack)
        actor_critic_networks = networks.make_actor_critic_networks_bin_pack(
            bin_pack=env.unwrapped,
            num_transformer_layers=cfg.env.network.num_transformer_layers,
            transformer_num_heads=cfg.env.network.transformer_num_heads,
            transformer_key_size=cfg.env.network.transformer_key_size,
            transformer_mlp_units=cfg.env.network.transformer_mlp_units,
        )
    elif cfg.env.name == "snake":
        assert isinstance(env.unwrapped, Snake)
        actor_critic_networks = networks.make_actor_critic_networks_snake(
            snake=env.unwrapped,
            num_channels=cfg.env.network.num_channels,
            policy_layers=cfg.env.network.policy_layers,
            value_layers=cfg.env.network.value_layers,
        )
    elif cfg.env.name == "tsp":
        assert isinstance(env.unwrapped, TSP)
        actor_critic_networks = networks.make_actor_critic_networks_tsp(
            tsp=env.unwrapped,
            transformer_num_blocks=cfg.env.network.transformer_num_blocks,
            transformer_num_heads=cfg.env.network.transformer_num_heads,
            transformer_key_size=cfg.env.network.transformer_key_size,
            transformer_mlp_units=cfg.env.network.transformer_mlp_units,
            mean_cities_in_query=cfg.env.network.mean_cities_in_query,
        )
    elif cfg.env.name == "knapsack":
        assert isinstance(env.unwrapped, Knapsack)
        actor_critic_networks = networks.make_actor_critic_networks_knapsack(
            knapsack=env.unwrapped,
            transformer_num_blocks=cfg.env.network.transformer_num_blocks,
            transformer_num_heads=cfg.env.network.transformer_num_heads,
            transformer_key_size=cfg.env.network.transformer_key_size,
            transformer_mlp_units=cfg.env.network.transformer_mlp_units,
        )
    elif cfg.env.name == "job_shop":
        assert isinstance(env.unwrapped, JobShop)
        actor_critic_networks = networks.make_actor_critic_networks_job_shop(
            job_shop=env.unwrapped,
            num_layers_machines=cfg.env.network.num_layers_machines,
            num_layers_operations=cfg.env.network.num_layers_operations,
            num_layers_joint_machines_jobs=cfg.env.network.num_layers_joint_machines_jobs,
            transformer_num_heads=cfg.env.network.transformer_num_heads,
            transformer_key_size=cfg.env.network.transformer_key_size,
            transformer_mlp_units=cfg.env.network.transformer_mlp_units,
        )
    elif cfg.env.name == "cvrp":
        assert isinstance(env.unwrapped, CVRP)
        actor_critic_networks = networks.make_actor_critic_networks_cvrp(
            cvrp=env.unwrapped,
            transformer_num_blocks=cfg.env.network.transformer_num_blocks,
            transformer_num_heads=cfg.env.network.transformer_num_heads,
            transformer_key_size=cfg.env.network.transformer_key_size,
            transformer_mlp_units=cfg.env.network.transformer_mlp_units,
            mean_nodes_in_query=cfg.env.network.mean_nodes_in_query,
        )
    elif cfg.env.name == "multi_cvrp":
        assert isinstance(env.unwrapped, MultiCVRP)
        actor_critic_networks = networks.make_actor_critic_networks_multicvrp(
            MultiCVRP=env.unwrapped,
            num_vehicles=cfg.env.network.num_vehicles,
            num_customers=cfg.env.network.num_customers,
            num_layers_vehicles=cfg.env.network.num_layers_vehicles,
            num_layers_customers=cfg.env.network.num_layers_customers,
            transformer_num_heads=cfg.env.network.transformer_num_heads,
            transformer_key_size=cfg.env.network.transformer_key_size,
            transformer_mlp_units=cfg.env.network.transformer_mlp_units,
        )
    elif cfg.env.name == "game_2048":
        assert isinstance(env.unwrapped, Game2048)
        actor_critic_networks = networks.make_actor_critic_networks_game_2048(
            game_2048=env.unwrapped,
            num_channels=cfg.env.network.num_channels,
            policy_layers=cfg.env.network.policy_layers,
            value_layers=cfg.env.network.value_layers,
        )
    elif cfg.env.name == "rubiks_cube":
        assert isinstance(env.unwrapped, RubiksCube)
        actor_critic_networks = networks.make_actor_critic_networks_rubiks_cube(
            rubiks_cube=env.unwrapped,
            cube_embed_dim=cfg.env.network.cube_embed_dim,
            step_count_embed_dim=cfg.env.network.step_count_embed_dim,
            dense_layer_dims=cfg.env.network.dense_layer_dims,
        )
    elif cfg.env.name == "sudoku":
        assert isinstance(env.unwrapped, Sudoku)
        actor_critic_networks = networks.make_equivariant_actor_critic_networks_sudoku(
            sudoku=env.unwrapped,
            num_heads=cfg.env.network.num_heads,
            key_size=cfg.env.network.key_size,
            policy_layers=cfg.env.network.policy_layers,
            value_layers=cfg.env.network.value_layers,
        )
    elif cfg.env.name == "robot_warehouse":
        assert isinstance(env.unwrapped, RobotWarehouse)
        actor_critic_networks = networks.make_actor_critic_networks_robot_warehouse(
            robot_warehouse=env.unwrapped,
            transformer_num_blocks=cfg.env.network.transformer_num_blocks,
            transformer_num_heads=cfg.env.network.transformer_num_heads,
            transformer_key_size=cfg.env.network.transformer_key_size,
            transformer_mlp_units=cfg.env.network.transformer_mlp_units,
        )
    elif cfg.env.name == "minesweeper":
        assert isinstance(env.unwrapped, Minesweeper)
        actor_critic_networks = networks.make_actor_critic_networks_minesweeper(
            minesweeper=env.unwrapped,
            board_embed_dim=cfg.env.network.board_embed_dim,
            board_conv_channels=cfg.env.network.board_conv_channels,
            board_kernel_shape=cfg.env.network.board_kernel_shape,
            num_mines_embed_dim=cfg.env.network.num_mines_embed_dim,
            final_layer_dims=cfg.env.network.final_layer_dims,
        )
    elif cfg.env.name == "maze":
        assert isinstance(env.unwrapped, Maze)
        actor_critic_networks = networks.make_actor_critic_networks_maze(
            maze=env.unwrapped,
            num_channels=cfg.env.network.num_channels,
            policy_layers=cfg.env.network.policy_layers,
            value_layers=cfg.env.network.value_layers,
        )
    elif cfg.env.name == "cleaner":
        assert isinstance(env.unwrapped, Cleaner)
        actor_critic_networks = networks.make_actor_critic_networks_cleaner(
            cleaner=env.unwrapped,
            num_conv_channels=cfg.env.network.num_conv_channels,
            policy_layers=cfg.env.network.policy_layers,
            value_layers=cfg.env.network.value_layers,
        )
    elif cfg.env.name == "connector":
        assert isinstance(env.unwrapped, Connector)
        actor_critic_networks = networks.make_actor_critic_networks_connector(
            connector=env.unwrapped,
            transformer_num_blocks=cfg.env.network.transformer_num_blocks,
            transformer_num_heads=cfg.env.network.transformer_num_heads,
            transformer_key_size=cfg.env.network.transformer_key_size,
            transformer_mlp_units=cfg.env.network.transformer_mlp_units,
            conv_n_channels=cfg.env.network.conv_n_channels,
        )
    elif cfg.env.name == "tetris":
        assert isinstance(env.unwrapped, Tetris)
        actor_critic_networks = networks.make_actor_critic_networks_tetris(
            tetris=env.unwrapped,
            conv_num_channels=cfg.env.network.conv_num_channels,
            tetromino_layers=cfg.env.network.tetromino_layers,
            head_layers=cfg.env.network.head_layers,
        )
    elif cfg.env.name == "mmst":
        assert isinstance(env.unwrapped, MMST)
        actor_critic_networks = networks.make_actor_critic_networks_mmst(
            mmst=env.unwrapped,
            num_transformer_layers=cfg.env.network.num_transformer_layers,
            transformer_num_heads=cfg.env.network.transformer_num_heads,
            transformer_key_size=cfg.env.network.transformer_key_size,
            transformer_mlp_units=cfg.env.network.transformer_mlp_units,
        )
    elif cfg.env.name == "graph_coloring":
        assert isinstance(env.unwrapped, GraphColoring)
        actor_critic_networks = networks.make_actor_critic_networks_graph_coloring(
            graph_coloring=env.unwrapped,
            num_transformer_layers=cfg.env.network.num_transformer_layers,
            transformer_num_heads=cfg.env.network.transformer_num_heads,
            transformer_key_size=cfg.env.network.transformer_key_size,
            transformer_mlp_units=cfg.env.network.transformer_mlp_units,
        )
    else:
        raise ValueError(f"Environment name not found. Got {cfg.env.name}.")
    return actor_critic_networks


def setup_evaluators(cfg: DictConfig, agent: Agent) -> Tuple[Evaluator, Evaluator]:
    env = _make_raw_env(cfg)
    stochastic_eval = Evaluator(
        eval_env=env,
        agent=agent,
        total_batch_size=cfg.env.evaluation.eval_total_batch_size,
        stochastic=True,
    )
    greedy_eval = Evaluator(
        eval_env=env,
        agent=agent,
        total_batch_size=cfg.env.evaluation.greedy_eval_total_batch_size,
        stochastic=False,
    )
    return stochastic_eval, greedy_eval


def setup_training_state(
    env: Environment, agent: Agent, key: chex.PRNGKey
) -> TrainingState:
    params_key, reset_key, acting_key = jax.random.split(key, 3)

    # Initialize params.
    params_state = agent.init_params(params_key)

    # Initialize environment states.
    num_local_devices = jax.local_device_count()
    num_global_devices = jax.device_count()
    num_workers = num_global_devices // num_local_devices
    local_batch_size = agent.total_batch_size // num_global_devices
    reset_keys = jax.random.split(reset_key, agent.total_batch_size).reshape(
        (
            num_workers,
            num_local_devices,
            local_batch_size,
            -1,
        )
    )
    reset_keys_per_worker = reset_keys[jax.process_index()]
    env_state, timestep = jax.pmap(env.reset, axis_name="devices")(
        reset_keys_per_worker
    )

    # Initialize acting states.
    acting_key_per_device = jax.random.split(acting_key, num_global_devices).reshape(
        num_workers, num_local_devices, -1
    )
    acting_key_per_worker_device = acting_key_per_device[jax.process_index()]
    acting_state = ActingState(
        state=env_state,
        timestep=timestep,
        key=acting_key_per_worker_device,
        episode_count=jnp.zeros(num_local_devices, float),
        env_step_count=jnp.zeros(num_local_devices, float),
    )

    # Build the training state.
    training_state = TrainingState(
        params_state=jax.device_put_replicated(params_state, jax.local_devices()),
        acting_state=acting_state,
    )
    return training_state
