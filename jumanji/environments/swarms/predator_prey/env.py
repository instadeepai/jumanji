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

from functools import cached_property
from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from esquilax.transforms import nearest_neighbour, spatial

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.swarms.common.types import AgentParams
from jumanji.environments.swarms.common.updates import init_state, update_state, view
from jumanji.types import TimeStep, restart, transition

from .types import Actions, Observation, Rewards, State
from .updates import (
    distance_predator_rewards,
    distance_prey_rewards,
    sparse_predator_rewards,
    sparse_prey_rewards,
)


class PredatorPrey(Environment):
    def __init__(
        self,
        num_predators: int,
        num_prey: int,
        prey_vision_range: float,
        predator_vision_range: float,
        num_vision: int,
        agent_radius: float,
        sparse_rewards: bool,
        prey_penalty: float,
        predator_rewards: float,
        predator_max_rotate: float,
        predator_max_accelerate: float,
        predator_min_speed: float,
        predator_max_speed: float,
        predator_view_angle: float,
        prey_max_rotate: float,
        prey_max_accelerate: float,
        prey_min_speed: float,
        prey_max_speed: float,
        prey_view_angle: float,
    ) -> None:
        self.num_predators = num_predators
        self.num_prey = num_prey
        self.prey_vision_range = prey_vision_range
        self.predator_vision_range = predator_vision_range
        self.num_vision = num_vision
        self.agent_radius = agent_radius
        self.sparse_rewards = sparse_rewards
        self.prey_penalty = prey_penalty
        self.predator_rewards = predator_rewards
        self.predator_params = AgentParams(
            max_rotate=predator_max_rotate,
            max_accelerate=predator_max_accelerate,
            min_speed=predator_min_speed,
            max_speed=predator_max_speed,
            view_angle=predator_view_angle,
        )
        self.prey_params = AgentParams(
            max_rotate=prey_max_rotate,
            max_accelerate=prey_max_accelerate,
            min_speed=prey_min_speed,
            max_speed=prey_max_speed,
            view_angle=prey_view_angle,
        )
        super().__init__()

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Predator-prey flock environment:",
                f" - num predators: {self.num_predators}",
                f" - num prey: {self.num_prey}",
                f" - prey vision range: {self.prey_vision_range}",
                f" - predator vision range: {self.predator_vision_range}"
                f" - num vision: {self.num_vision}"
                f" - agent radius: {self.agent_radius}"
                f" - sparse-rewards: {self.sparse_rewards}",
                f" - prey-penalty: {self.prey_penalty}",
                f" - predator-rewards: {self.predator_rewards}",
            ]
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment to an initial state.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: State object corresponding to the new state of the environment,
            timestep: TimeStep object corresponding the first timestep returned by the environment,
        """
        key, predator_key, prey_key = jax.random.split(key, num=3)
        predator_state = init_state(
            self.num_predators, self.predator_params, predator_key
        )
        prey_state = init_state(self.num_prey, self.prey_params, prey_key)
        state = State(predators=predator_state, prey=prey_state, key=key)
        timestep = restart(observation=self._state_to_observation(state))
        return state, timestep

    def step(
        self, state: State, action: Actions
    ) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the action to take.

        Returns:
            state: State object corresponding to the next state of the environment,
            timestep: TimeStep object corresponding the timestep returned by the environment,
        """
        predators = update_state(
            state.key, self.predator_params, state.predators, action.predators
        )
        prey = update_state(state.key, self.prey_params, state.prey, action.prey)

        state = State(
            predators=predators,
            prey=prey,
            key=state.key,
        )

        if self.sparse_rewards:
            rewards = self.state_to_sparse_rewards(state)
        else:
            rewards = self.state_to_distance_rewards(state)

        observation = self._state_to_observation(state)
        timestep = transition(rewards, observation)
        return state, timestep

    def _state_to_observation(self, state: State) -> Observation:

        prey_obs_predators = spatial(
            view,
            reduction=jnp.minimum,
            default=jnp.ones((self.num_vision,)),
            include_self=False,
            i_range=self.prey_vision_range,
        )(
            state.key,
            (self.prey_params.view_angle, self.agent_radius),
            state.prey,
            state.predators,
            pos=state.prey.pos,
            pos_b=state.predators.pos,
            n_view=self.num_vision,
            i_range=self.prey_vision_range,
        )
        prey_obs_prey = spatial(
            view,
            reduction=jnp.minimum,
            default=jnp.ones((self.num_vision,)),
            include_self=False,
            i_range=self.prey_vision_range,
        )(
            state.key,
            (self.predator_params.view_angle, self.agent_radius),
            state.prey,
            state.prey,
            pos=state.prey.pos,
            n_view=self.num_vision,
            i_range=self.prey_vision_range,
        )
        predator_obs_prey = spatial(
            view,
            reduction=jnp.minimum,
            default=jnp.ones((self.num_vision,)),
            include_self=False,
            i_range=self.predator_vision_range,
        )(
            state.key,
            (self.predator_params.view_angle, self.agent_radius),
            state.predators,
            state.prey,
            pos=state.predators.pos,
            pos_b=state.prey.pos,
            n_view=self.num_vision,
            i_range=self.predator_vision_range,
        )
        predator_obs_predator = spatial(
            view,
            reduction=jnp.minimum,
            default=jnp.ones((self.num_vision,)),
            include_self=False,
            i_range=self.predator_vision_range,
        )(
            state.key,
            (self.predator_params.view_angle, self.agent_radius),
            state.predators,
            state.predators,
            pos=state.predators.pos,
            n_view=self.num_vision,
            i_range=self.predator_vision_range,
        )

        predator_obs = jnp.hstack([predator_obs_prey, predator_obs_predator])
        prey_obs = jnp.hstack([prey_obs_predators, prey_obs_prey])

        return Observation(
            predators=predator_obs,
            prey=prey_obs,
        )

    def state_to_sparse_rewards(self, state: State) -> Rewards:
        prey_rewards = spatial(
            sparse_prey_rewards,
            reduction=jnp.add,
            default=0.0,
            include_self=False,
            i_range=2 * self.agent_radius,
        )(
            state.key,
            self.prey_penalty,
            None,
            None,
            pos=state.prey.pos,
            pos_b=state.predators.pos,
        )
        predator_rewards = nearest_neighbour(
            sparse_predator_rewards,
            default=0.0,
            i_range=2 * self.agent_radius,
        )(
            state.key,
            self.predator_rewards,
            None,
            None,
            pos=state.predators.pos,
            pos_b=state.prey.pos,
        )
        return Rewards(
            predators=predator_rewards,
            prey=prey_rewards,
        )

    def state_to_distance_rewards(self, state: State) -> Rewards:

        prey_rewards = spatial(
            distance_prey_rewards,
            reduction=jnp.add,
            default=0.0,
            include_self=False,
            i_range=self.prey_vision_range,
        )(
            state.key,
            self.prey_penalty,
            state.prey,
            state.predators,
            pos=state.prey.pos,
            pos_b=state.predators.pos,
            i_range=self.prey_vision_range,
        )
        predator_rewards = spatial(
            distance_predator_rewards,
            reduction=jnp.add,
            default=0.0,
            include_self=False,
            i_range=self.predator_vision_range,
        )(
            state.key,
            self.predator_rewards,
            state.predators,
            state.prey,
            pos=state.predators.pos,
            pos_b=state.prey.pos,
            i_range=self.prey_vision_range,
        )

        return Rewards(
            predators=predator_rewards,
            prey=prey_rewards,
        )

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec.

        Returns:
            observation_spec: a potentially nested `Spec` structure representing the observation.
        """
        predators = specs.BoundedArray(
            shape=(self.num_predators, 2 * self.num_vision),
            minimum=0.0,
            maximum=1.0,
            dtype=float,
            name="predators",
        )
        prey = specs.BoundedArray(
            shape=(self.num_prey, 2 * self.num_vision),
            minimum=0.0,
            maximum=1.0,
            dtype=float,
            name="prey",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            predators=predators,
            prey=prey,
        )

    @cached_property
    def action_spec(self) -> specs.Spec[Actions]:
        """Returns the action spec.

        Returns:
            action_spec: a potentially nested `Spec` structure representing the action.
        """
        predators = specs.BoundedArray(
            shape=(self.num_predators, 2),
            minimum=-1.0,
            maximum=1.0,
            dtype=float,
            name="predators",
        )
        prey = specs.BoundedArray(
            shape=(self.num_prey, 2),
            minimum=-1.0,
            maximum=1.0,
            dtype=float,
            name="prey",
        )
        return specs.Spec(
            Actions,
            "ActionSpec",
            predators=predators,
            prey=prey,
        )

    @cached_property
    def reward_spec(self) -> specs.Spec[Rewards]:  # type: ignore[override]
        """Returns the reward spec. By default, this is assumed to be a single float.

        Returns:
            reward_spec: a `specs.Array` spec.
        """
        predators = specs.Array(
            shape=(self.num_predators,),
            dtype=float,
            name="predators",
        )
        prey = specs.Array(
            shape=(self.num_prey,),
            dtype=float,
            name="prey",
        )
        return specs.Spec(
            Rewards,
            "rewardsSpec",
            predators=predators,
            prey=prey,
        )

    def render(self, state: State) -> Any:
        """Render frames of the environment for a given state.

        Args:
            state: State object containing the current dynamics of the environment.
        """
        raise NotImplementedError("Render method not implemented for this environment.")

    def close(self) -> None:
        """Perform any necessary cleanup."""
        pass
