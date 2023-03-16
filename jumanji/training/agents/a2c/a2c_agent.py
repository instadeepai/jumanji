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

import functools
from typing import Any, Callable, Dict, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import rlax

from jumanji.env import Environment
from jumanji.training.agents.base import Agent
from jumanji.training.networks.actor_critic import ActorCriticNetworks
from jumanji.training.types import (
    ActingState,
    ActorCriticParams,
    ParamsState,
    TrainingState,
    Transition,
)


class A2CAgent(Agent):
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
    ) -> None:
        super().__init__(total_batch_size=total_batch_size)
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

    def run_epoch(self, training_state: TrainingState) -> Tuple[TrainingState, Dict]:
        if not isinstance(training_state.params_state, ParamsState):
            raise TypeError(
                "Expected params_state to be of type ParamsState, got "
                f"type {type(training_state.params_state)}."
            )
        grad, (acting_state, metrics) = jax.grad(self.a2c_loss, has_aux=True)(
            training_state.params_state.params,
            training_state.acting_state,
        )
        grad, metrics = jax.lax.pmean((grad, metrics), "devices")
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
            acting_state=acting_state,
        )
        return training_state, metrics

    def a2c_loss(
        self,
        params: ActorCriticParams,
        acting_state: ActingState,
    ) -> Tuple[float, Tuple[ActingState, Dict]]:
        parametric_action_distribution = (
            self.actor_critic_networks.parametric_action_distribution
        )
        value_apply = self.actor_critic_networks.value_network.apply

        acting_state, data = self.rollout(
            policy_params=params.actor,
            acting_state=acting_state,
        )  # data.shape == (T, B, ...)
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
            parametric_action_distribution.entropy(data.logits, acting_state.key)
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
        return total_loss, (acting_state, metrics)

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
                episode_count=acting_state.episode_count
                + jax.lax.psum(next_timestep.last().sum(), "devices"),
                env_step_count=acting_state.env_step_count
                + jax.lax.psum(self.batch_size_per_device, "devices"),
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
        acting_state, data = jax.lax.scan(run_one_step, acting_state, acting_keys)
        return acting_state, data
