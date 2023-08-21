import functools
from typing import Any, Callable, Dict, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import rlax

from jumanji.env import Environment
from jumanji.training.networks.actor_critic import ActorCriticNetworks
from jumanji.training.types import (
    ActingState,
    ActorCriticParams,
    ParamsState,
    TrainingState,
    Transition,
)

# from jumanji.training.agents.a2c import a2c_agent


class A2CAgent:
    def __init__(
        self,
        env: Environment,
        n_steps: int,
        total_batch_size: int,
        actor_critic_networks: ActorCriticNetworks,
        optimizer: optax.GradientTransformation,
        normalize_advantage: bool,
        discount_factor: float,
        bootstrapping_factor: float,
        l_pg: float,
        l_td: float,
        l_en: float,
        gpu_acting: bool
    ) -> None:
        super().__init__()
        self.gpu_acting = gpu_acting
        # Hack from Agent class
        self.total_batch_size = total_batch_size
        num_devices = jax.local_device_count()
        assert total_batch_size % num_devices == 0, (
            "The total batch size must be a multiple of the number of devices, "
            f"got total_batch_size={total_batch_size} and num_devices={num_devices}."
        )
        self.batch_size_per_device = total_batch_size // num_devices

        # Back to classic agent init without hack
        self.env = env
        self.observation_spec = env.observation_spec()
        self.n_steps = n_steps
        self.actor_critic_networks = actor_critic_networks
        self.optimizer = optimizer
        self.normalize_advantage = normalize_advantage
        self.discount_factor = discount_factor
        self.bootstrapping_factor = bootstrapping_factor
        self.l_pg = l_pg
        self.l_td = l_td
        self.l_en = l_en

    def init_params(self, key: chex.PRNGKey) -> ParamsState:
        actor_key, critic_key = jax.random.split(key)
        dummy_obs = jax.tree_util.tree_map(
            lambda x: x[None, ...], self.observation_spec.generate_value()
        )  # Add batch dim
        params = ActorCriticParams(
            actor=self.actor_critic_networks.policy_network.init(actor_key, dummy_obs),
            critic=self.actor_critic_networks.value_network.init(critic_key, dummy_obs),
        )
        opt_state = self.optimizer.init(params)
        params_state = ParamsState(
            params=params,
            opt_state=opt_state,
            update_count=jnp.array(0, float),
        )
        return params_state

    def gradient_step(self,
                  training_state: TrainingState,
                  data: Transition,
                  ) -> Tuple[TrainingState, Dict]:
        if not isinstance(training_state.params_state, ParamsState):
            raise TypeError(
                "Expected params_state to be of type ParamsState, got "
                f"type {type(training_state.params_state)}."
            )
        grad, metrics = jax.grad(self.a2c_loss, has_aux=True)(
            training_state.params_state.params,
            data,
            training_state.acting_state.key
        )
        # grad, metrics = jax.lax.pmean((grad, metrics), "devices")
        updates, opt_state = self.optimizer.update(
            grad, training_state.params_state.opt_state
        )
        params = optax.apply_updates(training_state.params_state.params, updates)
        training_state = TrainingState(
            params_state=ParamsState(
                params=params,
                opt_state=opt_state,
                update_count=training_state.params_state.update_count + 1,
            ),
            acting_state=training_state.acting_state,
        )
        return training_state, metrics

    def a2c_loss(
        self,
        params: ActorCriticParams,
        data: Transition,
        key: chex.PRNGKey
    ) -> Tuple[float, Tuple[ActingState, Dict]]:
        parametric_action_distribution = (
            self.actor_critic_networks.parametric_action_distribution
        )
        value_apply = self.actor_critic_networks.value_network.apply


        last_observation = jax.tree_util.tree_map(
            lambda x: x[-1], data.next_observation
        )
        observation = jax.tree_util.tree_map(
            lambda obs_0_tm1, obs_t: jnp.concatenate([obs_0_tm1, obs_t[None]], axis=0),
            data.observation,
            last_observation,
        )

        value = jax.vmap(value_apply, in_axes=(None, 0))(params.critic, observation)
        discounts = jnp.asarray(self.discount_factor * data.discount, float)
        value_tm1 = value[:-1]
        value_t = value[1:]
        advantage = jax.vmap(
            functools.partial(
                rlax.td_lambda,
                lambda_=self.bootstrapping_factor,
                stop_target_gradients=True,
            ),
            in_axes=1,
            out_axes=1,
        )(
            value_tm1,
            data.reward,
            discounts,
            value_t,
        )

        # Compute the critic loss before potentially normalizing the advantages.
        critic_loss = jnp.mean(advantage**2)

        # Compute the policy loss with optional advantage normalization.
        metrics: Dict = {}
        if self.normalize_advantage:
            metrics.update(unnormalized_advantage=jnp.mean(advantage))
            advantage = jax.nn.standardize(advantage)
        policy_loss = -jnp.mean(jax.lax.stop_gradient(advantage) * data.log_prob)

        # Compute the entropy loss, i.e. negative of the entropy.
        entropy = jnp.mean(
            parametric_action_distribution.entropy(data.logits, key)
        )
        entropy_loss = -entropy

        total_loss = (
            self.l_pg * policy_loss + self.l_td * critic_loss + self.l_en * entropy_loss
        )
        metrics.update(
            total_loss=total_loss,
            policy_loss=policy_loss,
            critic_loss=critic_loss,
            entropy_loss=entropy_loss,
            entropy=entropy,
            advantage=jnp.mean(advantage),
            value=jnp.mean(value),
        )
        if data.extras:
            metrics.update(data.extras)
        return total_loss, metrics

    def make_policy(
        self,
        policy_params: hk.Params,
        stochastic: bool = True,
    ) -> Callable[
        [Any, chex.PRNGKey], Tuple[chex.Array, Tuple[chex.Array, chex.Array]]
    ]:
        policy_network = self.actor_critic_networks.policy_network
        parametric_action_distribution = (
            self.actor_critic_networks.parametric_action_distribution
        )

        def policy(
            observation: Any, key: chex.PRNGKey
        ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
            logits = policy_network.apply(policy_params, observation)
            if stochastic:
                raw_action = parametric_action_distribution.sample_no_postprocessing(
                    logits, key
                )
                log_prob = parametric_action_distribution.log_prob(logits, raw_action)
            else:
                del key
                raw_action = parametric_action_distribution.mode_no_postprocessing(
                    logits
                )
                # log_prob is log(1), i.e. 0, for a greedy policy (deterministic distribution).
                log_prob = jnp.zeros_like(
                    parametric_action_distribution.log_prob(logits, raw_action)
                )
            action = parametric_action_distribution.postprocess(raw_action)
            return action, (log_prob, logits)

        return policy

    def rollout(
        self,
        policy_params: hk.Params,
        acting_state: ActingState,
    ) -> Tuple[ActingState, Transition]:
        """Rollout for training purposes.
        Returns:
            shape (n_steps, batch_size_per_device, *)
        """

        def run_one_step(
            acting_state: ActingState, key: chex.PRNGKey, policy_params: hk.Params,
        ) -> Tuple[ActingState, Transition]:
            timestep = acting_state.timestep
            policy = self.make_policy(policy_params=policy_params, stochastic=True)
            action, (log_prob, logits) = policy(timestep.observation, key)
            next_env_state, next_timestep = self.env.step(acting_state.state, action)

            acting_state = ActingState(
                state=next_env_state,
                timestep=next_timestep,
                key=key,
                episode_count=acting_state.episode_count + next_timestep.last().sum(),
                # + jax.lax.psum(next_timestep.last().sum(), "devices"),
                env_step_count=acting_state.env_step_count + self.batch_size_per_device,
                # + jax.lax.psum(self.batch_size_per_device, "devices"),
            )

            transition = Transition(
                observation=timestep.observation,
                action=action,
                reward=next_timestep.reward,
                discount=next_timestep.discount,
                next_observation=next_timestep.observation,
                log_prob=log_prob,
                logits=logits,
                extras=next_timestep.extras,
            )

            return acting_state, transition

        acting_keys = jax.random.split(acting_state.key, self.n_steps).reshape(
            (self.n_steps, -1)
        )

        datas = []
        for i in range(self.n_steps):
            acting_state, data = jax.jit(run_one_step)(acting_state, acting_keys[i], policy_params)
            datas.append(data)

        def func(args):
            return jnp.stack(args)

        data = jax.tree_map(lambda *xs: func(xs), *datas)
        return acting_state, data

    # def rollout(
    #     self,
    #     policy_params: hk.Params,
    #     acting_state: ActingState,
    # ) -> Tuple[ActingState, Transition]:
    #     if self.gpu_acting:
    #         return self.rollout_gpu(policy_params, acting_state)
    #     else:
    #         return self.rollout_cpu(policy_params, acting_state)

    def rollout_gpu(
        self,
        policy_params: hk.Params,
        acting_state: ActingState,
    ) -> Tuple[ActingState, Transition]:
        """Rollout for training purposes.
        Returns:
            shape (n_steps, batch_size_per_device, *)
        """
        policy = self.make_policy(policy_params=policy_params, stochastic=True)

        def run_one_step(
            acting_state: ActingState, key: chex.PRNGKey
        ) -> Tuple[ActingState, Transition]:
            timestep = acting_state.timestep
            action, (log_prob, logits) = policy(timestep.observation, key)
            next_env_state, next_timestep = self.env.step(acting_state.state, action)

            acting_state = ActingState(
                state=next_env_state,
                timestep=next_timestep,
                key=key,
                episode_count=acting_state.episode_count + next_timestep.last().sum(),
                # + jax.lax.psum(next_timestep.last().sum(), "devices"),
                env_step_count=acting_state.env_step_count + self.batch_size_per_device,
                # + jax.lax.psum(self.batch_size_per_device, "devices"),
            )

            transition = Transition(
                observation=timestep.observation,
                action=action,
                reward=next_timestep.reward,
                discount=next_timestep.discount,
                next_observation=next_timestep.observation,
                log_prob=log_prob,
                logits=logits,
                extras=next_timestep.extras,
            )

            return acting_state, transition

        acting_keys = jax.random.split(acting_state.key, self.n_steps).reshape(
            (self.n_steps, -1)
        )

        datas = []
        for i in range(self.n_steps):
            acting_state, data = jax.jit(run_one_step)(acting_state, acting_keys[i])
            datas.append(data)

        def func(args):
            return jnp.stack(args)

        data = jax.tree_map(lambda *xs: func(xs), *datas)
        return acting_state, data

    def rollout_cpu(
        self,
        policy_params: hk.Params,
        acting_state: ActingState,
    ) -> Tuple[ActingState, Transition]:
        """Rollout for training purposes.
        Returns:
            shape (n_steps, batch_size_per_device, *)
        """
        policy_params, acting_state = jax.device_put((policy_params, acting_state), device=jax.devices("cpu")[0])


        policy = self.make_policy(policy_params=policy_params, stochastic=True)

        def run_one_step(
            acting_state: ActingState, key: chex.PRNGKey
        ) -> Tuple[ActingState, Transition]:
            timestep = acting_state.timestep
            action, (log_prob, logits) = policy(timestep.observation, key)
            next_env_state, next_timestep = self.env.step(acting_state.state, action)

            acting_state = ActingState(
                state=next_env_state,
                timestep=next_timestep,
                key=key,
                episode_count=acting_state.episode_count + next_timestep.last().sum(),
                # + jax.lax.psum(next_timestep.last().sum(), "devices"),
                env_step_count=acting_state.env_step_count + self.batch_size_per_device,
                # + jax.lax.psum(self.batch_size_per_device, "devices"),
            )

            transition = Transition(
                observation=timestep.observation,
                action=action,
                reward=next_timestep.reward,
                discount=next_timestep.discount,
                next_observation=next_timestep.observation,
                log_prob=log_prob,
                logits=logits,
                extras=next_timestep.extras,
            )

            return acting_state, transition

        acting_keys = jax.random.split(acting_state.key, self.n_steps).reshape(
            (self.n_steps, -1)
        )

        datas = []
        for i in range(self.n_steps):
            acting_state, data = jax.jit(run_one_step)(acting_state, acting_keys[i])
            datas.append(data)

        def func(args):
            return jnp.stack(args)

        data = jax.tree_map(lambda *xs: func(xs), *datas)
        acting_state, data = jax.device_put((acting_state, data), device=jax.devices()[0])
        return acting_state, data


    def rollout_cpu_v1(
        self,
        policy_params: hk.Params,
        acting_state: ActingState,
    ) -> Tuple[ActingState, Transition]:
        """Rollout for training purposes.
        Returns:
            shape (n_steps, batch_size_per_device, *)
        """

        @jax.jit
        def policy_forward(policy_params, observation, key):
            policy = self.make_policy(policy_params=policy_params, stochastic=True)
            return policy(observation, key)

        def run_one_step(
            acting_state: ActingState, key: chex.PRNGKey
        ) -> Tuple[ActingState, Transition]:
            timestep = acting_state.timestep
            action, (log_prob, logits) = policy_forward(policy_params, timestep.observation, key)
            device = jax.devices("cpu")[0]
            env_state, action = jax.device_put((acting_state.state, action), device=device)
            next_env_state, next_timestep = jax.jit(self.env.step)(env_state, action)
            next_env_state, next_timestep, acting_state, action = jax.device_put((next_env_state, next_timestep,
                                                                                  acting_state, action),
                                                                         device=jax.devices()[0])

            acting_state = ActingState(
                state=next_env_state,
                timestep=next_timestep,
                key=key,
                episode_count=acting_state.episode_count + next_timestep.last().sum(),
                # + jax.lax.psum(next_timestep.last().sum(), "devices"),
                env_step_count=acting_state.env_step_count + self.batch_size_per_device,
                # + jax.lax.psum(self.batch_size_per_device, "devices"),
            )

            transition = Transition(
                observation=timestep.observation,
                action=action,
                reward=next_timestep.reward,
                discount=next_timestep.discount,
                next_observation=next_timestep.observation,
                log_prob=log_prob,
                logits=logits,
                extras=next_timestep.extras,
            )

            return acting_state, transition

        acting_keys = jax.random.split(acting_state.key, self.n_steps).reshape(
            (self.n_steps, -1)
        )

        datas = []
        for i in range(self.n_steps):
            acting_state, data = run_one_step(acting_state, acting_keys[i])

            datas.append(data)

        def func(args):
            return jnp.stack(args)

        data = jax.tree_map(lambda *xs: func(xs), *datas)
        return acting_state, data


if __name__ == '__main__':
    # Note: we can use `n_step` = batch_size * `n_step` and batch size of 1 to get environment with no-parellelelizm.


    from train import train
    from hydra import compose, initialize

    import os
    import requests


    env = "rubiks_cube"  # @param ['bin_pack', 'cleaner', 'connector', 'cvrp', 'game_2048', 'graph_coloring', 'job_shop', 'knapsack', 'maze', 'minesweeper', 'mmst', 'multi_cvrp', 'robot_warehouse', 'rubiks_cube', 'snake', 'sudoku', 'tetris', 'tsp']
    agent = "a2c"  # @param ['random', 'a2c']
    batch_size = 32
    gpu_acting = False

    def download_file(url: str, file_path: str) -> None:
        # Send an HTTP GET request to the URL
        response = requests.get(url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            print("Failed to download the file.")


    os.makedirs("configs", exist_ok=True)
    config_url = "https://raw.githubusercontent.com/instadeepai/jumanji/main/jumanji/training/configs/config.yaml"
    download_file(config_url, "configs/config.yaml")
    env_url = f"https://raw.githubusercontent.com/instadeepai/jumanji/main/jumanji/training/configs/env/{env}.yaml"
    os.makedirs("configs/env", exist_ok=True)
    download_file(env_url, f"configs/env/{env}.yaml")


    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="config.yaml",
                      overrides=[f"env={env}", f"agent={agent}", "logger.type=terminal", "logger.save_checkpoint=true",
                                 "env.training.total_batch_size=16"])
        cfg.env.network.dense_layer_dims = [16, ]

    train(cfg, gpu_acting=gpu_acting)

