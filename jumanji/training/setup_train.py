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
import optax
from omegaconf import DictConfig

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments import (
    CVRP,
    TSP,
    BinPack,
    Connect4,
    JobShop,
    Knapsack,
    Minesweeper,
    Routing,
    RubiksCube,
    Snake,
)
from jumanji.training import networks
from jumanji.training.agents.a2c import A2CAgent
from jumanji.training.agents.base import Agent
from jumanji.training.agents.random import RandomAgent
from jumanji.training.evaluator import Evaluator
from jumanji.training.loggers import Logger, NeptuneLogger, TerminalLogger
from jumanji.training.networks.actor_critic import ActorCriticNetworks
from jumanji.training.networks.protocols import RandomPolicy
from jumanji.training.types import ActingState, TrainingState
from jumanji.wrappers import MultiToSingleWrapper, VmapAutoResetWrapper

ENV_FACTORY = {
    "binpack": BinPack,
    "cvrp": CVRP,
    "tsp": TSP,
    "connect4": Connect4,
    "snake": Snake,
    "routing": Routing,
    "rubiks_cube": RubiksCube,
    "minesweeper": Minesweeper,
    "knapsack": Knapsack,
    "jobshop": JobShop,
}


def setup_logger(cfg: DictConfig) -> Logger:
    logger: Logger
    if cfg.logger.type == "neptune":
        logger = NeptuneLogger(
            name=cfg.logger.name, project="InstaDeep/jumanji", cfg=cfg
        )
    elif cfg.logger.type == "terminal":
        logger = TerminalLogger(name=cfg.logger.name)
    else:
        raise ValueError(
            f"logger expected in ['neptune', 'terminal'], got {cfg.logger}."
        )
    return logger


def _make_raw_env(cfg: DictConfig) -> Environment:
    env_name = cfg.environment.name
    env_kwargs = cfg.environment.env_kwargs
    if "instance_generator_kwargs" in cfg.environment.keys():
        instance_generator_kwargs = cfg.environment.instance_generator_kwargs
        env_kwargs.update(instance_generator_kwargs or {})
    env: Environment = ENV_FACTORY[env_name](**env_kwargs)
    if isinstance(env.action_spec(), specs.MultiDiscreteArray):
        env = MultiToSingleWrapper(env)
    return env


def setup_env(cfg: DictConfig) -> Environment:
    env = _make_raw_env(cfg)
    env = VmapAutoResetWrapper(env)
    return env


def setup_agent(cfg: DictConfig, env: Environment) -> Agent:
    agent: Agent
    if cfg.agent.name == "random":
        random_policy = _setup_random_policy(cfg, env)
        agent = RandomAgent(
            env=env,
            n_steps=cfg.n_steps,
            total_batch_size=cfg.total_batch_size,
            random_policy=random_policy,
        )
    elif cfg.agent.name == "a2c":
        actor_critic_networks = _setup_actor_critic_neworks(cfg, env)
        optimizer = optax.adam(cfg.agent.learning_rate)
        agent = A2CAgent(
            env=env,
            n_steps=cfg.n_steps,
            total_batch_size=cfg.total_batch_size,
            actor_critic_networks=actor_critic_networks,
            optimizer=optimizer,
            normalize_advantage=cfg.agent.normalize_advantage,
            discount_factor=cfg.agent.discount_factor,
            bootstrapping_factor=cfg.agent.bootstrapping_factor,
            l_pg=cfg.agent.l_pg,
            l_td=cfg.agent.l_td,
            l_en=cfg.agent.l_en,
        )
    else:
        raise ValueError(
            f"Expected agent name to be in ['random', 'a2c'], got {cfg.agent.name}."
        )
    return agent


def _setup_random_policy(cfg: DictConfig, env: Environment) -> RandomPolicy:
    assert cfg.agent.network == "random"
    if cfg.environment.name == "binpack":
        assert isinstance(env.unwrapped, BinPack)
        random_policy = networks.make_random_policy_binpack(binpack=env.unwrapped)
    elif cfg.environment.name == "snake":
        assert isinstance(env.unwrapped, Snake)
        random_policy = networks.make_random_policy_snake()
    elif cfg.environment.name == "tsp":
        assert isinstance(env.unwrapped, TSP)
        random_policy = networks.make_random_policy_tsp()
    elif cfg.environment.name == "knapsack":
        assert isinstance(env.unwrapped, Knapsack)
        random_policy = networks.make_random_policy_knapsack()
    elif cfg.environment.name == "jobshop":
        assert isinstance(env.unwrapped, JobShop)
        random_policy = networks.make_random_policy_jobshop()
    elif cfg.environment.name == "cvrp":
        assert isinstance(env.unwrapped, CVRP)
        random_policy = networks.make_random_policy_cvrp()
    elif cfg.environment.name == "routing":
        assert isinstance(env.unwrapped, Routing)
        random_policy = networks.make_random_policy_routing(routing=env.unwrapped)
    elif cfg.environment.name == "rubiks_cube":
        assert isinstance(env.unwrapped, RubiksCube)
        random_policy = networks.make_random_policy_rubiks_cube(
            rubiks_cube=env.unwrapped
        )
    elif cfg.environment.name == "minesweeper":
        assert isinstance(env.unwrapped, Minesweeper)
        random_policy = networks.make_random_policy_minesweeper(
            minesweeper=env.unwrapped
        )
    else:
        raise ValueError(f"Environment name not found. Got {cfg.environment.name}.")
    return random_policy


def _setup_actor_critic_neworks(
    cfg: DictConfig, env: Environment
) -> ActorCriticNetworks:
    assert cfg.agent.network == "actor_critic"
    if cfg.environment.name == "binpack":
        assert isinstance(env.unwrapped, BinPack)
        actor_critic_networks = networks.make_actor_critic_networks_binpack(
            observation_spec=env.unwrapped.observation_spec(),
            policy_layers=cfg.environment.network.policy_layers,
            value_layers=cfg.environment.network.value_layers,
            transformer_n_blocks=cfg.environment.network.transformer_n_blocks,
            transformer_mlp_units=cfg.environment.network.transformer_mlp_units,
            transformer_key_size=cfg.environment.network.transformer_key_size,
            transformer_num_heads=cfg.environment.network.transformer_num_heads,
        )
    elif cfg.environment.name == "snake":
        assert isinstance(env.unwrapped, Snake)
        actor_critic_networks = networks.make_actor_critic_networks_snake(
            num_channels=cfg.environment.network.num_channels,
            policy_layers=cfg.environment.network.policy_layers,
            value_layers=cfg.environment.network.value_layers,
        )
    elif cfg.environment.name == "tsp":
        assert isinstance(env.unwrapped, TSP)
        actor_critic_networks = networks.make_actor_critic_networks_tsp(
            tsp=env.unwrapped,
            encoder_num_layers=cfg.environment.network.encoder_num_layers,
            encoder_num_heads=cfg.environment.network.encoder_num_heads,
            encoder_key_size=cfg.environment.network.encoder_key_size,
            encoder_model_size=cfg.environment.network.encoder_model_size,
            encoder_expand_factor=cfg.environment.network.encoder_expand_factor,
            decoder_num_heads=cfg.environment.network.decoder_num_heads,
            decoder_key_size=cfg.environment.network.decoder_key_size,
            decoder_model_size=cfg.environment.network.decoder_model_size,
        )
    elif cfg.environment.name == "knapsack":
        assert isinstance(env.unwrapped, Knapsack)
        actor_critic_networks = networks.make_actor_critic_networks_knapsack(
            knapsack=env.unwrapped,
            encoder_num_layers=cfg.environment.network.encoder_num_layers,
            encoder_num_heads=cfg.environment.network.encoder_num_heads,
            encoder_key_size=cfg.environment.network.encoder_key_size,
            encoder_model_size=cfg.environment.network.encoder_model_size,
            encoder_expand_factor=cfg.environment.network.encoder_expand_factor,
            decoder_num_heads=cfg.environment.network.decoder_num_heads,
            decoder_key_size=cfg.environment.network.decoder_key_size,
            decoder_model_size=cfg.environment.network.decoder_model_size,
        )
    elif cfg.environment.name == "jobshop":
        assert isinstance(env.unwrapped, JobShop)
        actor_critic_networks = networks.make_actor_critic_networks_jobshop(
            jobshop=env.unwrapped,
            policy_layers=cfg.environment.network.policy_layers,
            value_layers=cfg.environment.network.value_layers,
            operations_layers=cfg.environment.network.operations_layers,
            machines_layers=cfg.environment.network.machines_layers,
        )

    elif cfg.environment.name == "cvrp":
        assert isinstance(env.unwrapped, CVRP)
        actor_critic_networks = networks.make_actor_critic_networks_cvrp(
            cvrp=env.unwrapped,
            encoder_num_layers=cfg.environment.network.encoder_num_layers,
            encoder_num_heads=cfg.environment.network.encoder_num_heads,
            encoder_key_size=cfg.environment.network.encoder_key_size,
            encoder_model_size=cfg.environment.network.encoder_model_size,
            encoder_expand_factor=cfg.environment.network.encoder_expand_factor,
            decoder_num_heads=cfg.environment.network.decoder_num_heads,
            decoder_key_size=cfg.environment.network.decoder_key_size,
            decoder_model_size=cfg.environment.network.decoder_model_size,
        )
    elif cfg.environment.name == "routing":
        assert isinstance(env.unwrapped, Routing)
        actor_critic_networks = networks.make_actor_critic_networks_routing(
            routing=env.unwrapped,
            num_channels=cfg.environment.network.num_channels,
            policy_layers=cfg.environment.network.policy_layers,
            value_layers=cfg.environment.network.value_layers,
        )
    elif cfg.environment.name == "rubiks_cube":
        assert isinstance(env.unwrapped, RubiksCube)
        raise NotImplementedError
    elif cfg.environment.name == "minesweeper":
        assert isinstance(env.unwrapped, Minesweeper)
        raise NotImplementedError
    else:
        raise ValueError(f"Environment name not found. Got {cfg.environment.name}.")
    return actor_critic_networks


def setup_evaluator(cfg: DictConfig, agent: Agent) -> Evaluator:
    env = _make_raw_env(cfg)
    evaluator = Evaluator(
        eval_env=env,
        agent=agent,
        total_num_eval=cfg.total_num_eval,
    )
    return evaluator


def setup_training_state(
    env: Environment, agent: Agent, key: chex.PRNGKey
) -> TrainingState:
    params_key, reset_key, acting_key = jax.random.split(key, 3)

    # Initialize params.
    params_state = agent.init_params(params_key)

    # Initialize environment states.
    num_devices = jax.local_device_count()
    reset_keys = jax.random.split(reset_key, agent.total_batch_size).reshape(
        (num_devices, agent.batch_size_per_device, -1)
    )
    env_state, timestep = jax.pmap(env.reset, axis_name="devices")(reset_keys)

    # Initialize acting states.
    acting_key_per_device = jax.random.split(acting_key, num_devices)
    acting_state = ActingState(
        state=env_state,
        timestep=timestep,
        key=acting_key_per_device,
        episode_count=jnp.zeros(num_devices, jnp.float32),
        env_step_count=jnp.zeros(num_devices, jnp.float32),
    )

    # Build the training state.
    training_state = TrainingState(
        params_state=jax.device_put_replicated(params_state, jax.local_devices()),
        acting_state=acting_state,
    )
    return training_state
