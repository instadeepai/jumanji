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
from omegaconf import DictConfig

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments import CVRP, TSP, BinPack, Connect4, Knapsack, Routing, Snake
from jumanji.training import networks
from jumanji.training.agents.base import Agent
from jumanji.training.agents.random import RandomAgent, RandomPolicy
from jumanji.training.evaluator import Evaluator
from jumanji.training.loggers import Logger, TerminalLogger
from jumanji.training.types import ActingState, TrainingState
from jumanji.wrappers import AutoResetWrapper, MultiToSingleWrapper, VmapWrapper

ENV_FACTORY = {
    "binpack": BinPack,
    "cvrp": CVRP,
    "tsp": TSP,
    "connect4": Connect4,
    "snake": Snake,
    "routing": Routing,
    "knapsack": Knapsack,
}


def setup_logger(cfg: DictConfig) -> Logger:
    # TODO: output correct logger depending on cfg.
    logger = TerminalLogger()
    return logger


def _make_raw_env(cfg: DictConfig) -> Environment:
    env_name = cfg.environment.name
    env_kwargs = cfg.environment.env_kwargs
    if "instance_generator_kwargs" in cfg.environment.keys():
        env_kwargs.update(cfg.environment.instance_generator_kwargs)
    env: Environment = ENV_FACTORY[env_name](**env_kwargs)
    if isinstance(env.action_spec(), specs.MultiDiscreteArray):
        env = MultiToSingleWrapper(env)
    return env


def setup_env(cfg: DictConfig) -> Environment:
    env = _make_raw_env(cfg)
    env = AutoResetWrapper(env)
    env = VmapWrapper(env)
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
        raise NotImplementedError
        # TODO: uncomment when A2C is implemented.
        # actor_critic_networks = _setup_actor_critic_neworks(cfg, env)
        # optimizer = optax.adam(cfg.learning_rate)
        # agent = A2C(
        #     env=env,
        #     n_steps=cfg.n_steps,
        #     total_batch_size=cfg.total_batch_size,
        #     actor_critic_networks=actor_critic_networks,
        #     optimizer=optimizer,
        #     normalize_advantage=cfg.normalize_advantage,
        #     discount_factor=cfg.discount_factor,
        #     bootstrapping_factor=cfg.bootstrapping_factor,
        #     l_pg=cfg.agent.l_pg,
        #     l_td=cfg.agent.l_td,
        #     l_en=cfg.agent.l_en,
        # )
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
    elif cfg.environment.name == "cvrp":
        assert isinstance(env.unwrapped, CVRP)
        random_policy = networks.make_random_policy_cvrp()
    elif cfg.environment.name == "routing":
        assert isinstance(env.unwrapped, Routing)
        random_policy = networks.make_random_policy_routing(routing=env.unwrapped)
    else:
        raise ValueError(f"Environment name not found. Got {cfg.environment.name}.")
    return random_policy


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
