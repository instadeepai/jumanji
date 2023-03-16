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

from typing import Any, Callable, Dict, Optional, Tuple

import chex
import jax

from jumanji.env import Environment
from jumanji.training.agents.base import Agent
from jumanji.training.networks.protocols import RandomPolicy
from jumanji.training.types import ActingState, TrainingState


class RandomAgent(Agent):
    def __init__(
        self,
        env: Environment,
        n_steps: int,
        total_batch_size: int,
        random_policy: RandomPolicy,
    ) -> None:
        super().__init__(total_batch_size=total_batch_size)
        self.env = env
        self.observation_spec = env.observation_spec()
        self.n_steps = n_steps
        self.random_policy = random_policy

    def init_params(self, key: chex.PRNGKey) -> None:
        del key
        return None

    def run_epoch(self, training_state: TrainingState) -> Tuple[TrainingState, Dict]:
        acting_state, extras = self.random_rollout(
            acting_state=training_state.acting_state,
        )
        training_state = TrainingState(
            params_state=None,
            acting_state=acting_state,
        )
        metrics = {}
        if extras:
            metrics.update(extras)
        return training_state, metrics

    def make_policy(
        self,
        policy_params: None = None,
        stochastic: bool = True,
    ) -> Callable[[Any, chex.PRNGKey], chex.Array]:
        del stochastic
        return self.random_policy

    def random_rollout(
        self,
        acting_state: ActingState,
    ) -> Tuple[ActingState, Optional[Dict]]:
        """Rollout for training purposes.
        Returns:
            shape (n_steps, batch_size_per_device, *)
        """
        random_policy = self.make_policy()

        def run_one_step(
            acting_state: ActingState, key: chex.PRNGKey
        ) -> Tuple[ActingState, Optional[Dict]]:
            action = random_policy(acting_state.timestep.observation, key)
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
            extras = next_timestep.extras
            return acting_state, extras

        acting_keys = jax.random.split(acting_state.key, self.n_steps).reshape(
            (self.n_steps, -1)
        )
        acting_state, extras = jax.lax.scan(run_one_step, acting_state, acting_keys)
        return acting_state, extras
