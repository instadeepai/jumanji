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
from typing import Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
from esquilax.transforms import spatial
from matplotlib.animation import FuncAnimation

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.swarms.common.types import AgentParams
from jumanji.environments.swarms.common.updates import update_state, view
from jumanji.environments.swarms.predator_prey.generator import (
    Generator,
    RandomGenerator,
)
from jumanji.environments.swarms.predator_prey.rewards import DistanceRewards, RewardFn
from jumanji.environments.swarms.predator_prey.types import (
    Actions,
    Observation,
    Rewards,
    State,
)
from jumanji.environments.swarms.predator_prey.viewer import PredatorPreyViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class PredatorPrey(Environment):
    """A predator and prey flock environment

    Environment modelling two swarms of agent types, predators
    who are rewarded for avoiding pre agents, and conversely
    prey agent who are rewarded for touching/catching
    prey agents. Both agent types can consist of a large
    number of individual agents, each with individual (local)
    observations, actions, and rewards. Agents interact
    on a uniform space with wrapped boundaries.

    - observation: `Observation`
        Arrays representing each agent's local view of the environment.
        Each cell of the array represent the distance from the agent
        two the nearest other agents in the environment. Each agent type
        is observed independently.

        - predators: jax array (float) of shape (num_predators, 2 * num_vision)
        - prey: jax array (float) of shape (num_prey, 2 * num_vision)

    - action: `Actions`
        Arrays of individual agent actions. Each agents actions rotate and
        accelerate/decelerate the agent as [rotation, acceleration] on the range
        [-1, 1]. These values are then scaled to update agent velocities within
        given parameters.

        - predators: jax array (float) of shape (num_predators, 2)
        - prey: jax array (float) of shape (num_prey, 2)

    - reward: `Rewards`
        Arrays of individual agent rewards. Rewards generally depend on
        proximity to other agents, and so can vary dependent on
        density and agent radius and vision ranges.

        - predators: jax array (float) of shape (num_predators,)
        - prey: jax array (float) of shape (num_prey,)

    - state: `State`
        - predators: `AgentState`
            - pos: jax array (float) of shape (num_predators, 2) in the range [0, 1].
            - heading: jax array (float) of shape (num_predators,) in
                the range [0, 2pi].
            - speed: jax array (float) of shape (num_predators,) in the
                range [min_speed, max_speed].
        - prey: `AgentState`
            - pos: jax array (float) of shape (num_prey, 2) in the range [0, 1].
            - heading: jax array (float) of shape (num_prey,) in
                the range [0, 2pi].
            - speed: jax array (float) of shape (num_prey,) in the
                range [min_speed, max_speed].
        - key: jax array (uint32) of shape (2,)
        - step: int representing the current simulation step.


    ```python
    from jumanji.environments import PredatorPrey
    env = PredatorPrey(
        num_predators=2,
        num_prey=10,
        prey_vision_range=0.1,
        predator_vision_range=0.1,
        num_vision=10,
        agent_radius=0.01,
        sparse_rewards=True,
        prey_penalty=0.1,
        predator_rewards=0.2,
        predator_max_rotate=0.1,
        predator_max_accelerate=0.01,
        predator_min_speed=0.01,
        predator_max_speed=0.05,
        predator_view_angle=0.5,
        prey_max_rotate=0.1,
        prey_max_accelerate=0.01,
        prey_min_speed=0.01,
        prey_max_speed=0.05,
        prey_view_angle=0.5,
    )
    key = jax.random.PRNGKey(0)
    state, timestep = jax.jit(env.reset)(key)
    env.render(state)
    action = env.action_spec.generate_value()
    state, timestep = jax.jit(env.step)(state, action)
    env.render(state)
    ```
    """

    def __init__(
        self,
        num_predators: int,
        num_prey: int,
        prey_vision_range: float,
        predator_vision_range: float,
        num_vision: int,
        agent_radius: float,
        sparse_rewards: bool,
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
        max_steps: int = 10_000,
        viewer: Optional[Viewer[State]] = None,
        generator: Optional[Generator] = None,
        reward_fn: Optional[RewardFn] = None,
    ) -> None:
        """Instantiates a `PredatorPrey` environment

        Note:
            The environment is square with dimensions
            `[1.0, 1.0]` so parameters should be scaled
            appropriately. Also note that performance is
            dependent on agent vision and interaction ranges,
            where larger values can lead to large number of
            agent interactions.

        Args:
            num_predators: Number of predator agents.
            num_prey: Number of prey agents.
            prey_vision_range: Prey agent vision range.
            predator_vision_range: Predator agent vision range.
            num_vision: Number of cells/subdivisions in agent
            view models. Larger numbers provide a more accurate
                view, at the cost of the environment, at the cost
                of performance and memory usage.
            agent_radius: Radius of individual agents. This
                effects both agent collision range and how
                large they appear to other agents.
            sparse_rewards: If `True` fix rewards will be applied
                when agents are within a fixed collision range. If
                `False` rewards are dependent on distance to
                other agents with vision range.
            predator_max_rotate: Maximum rotation predator agents can
                turn within a step. Should be a value from [0,1]
                representing a fraction of pi radians.
            predator_max_accelerate: Maximum acceleration/deceleration
                a predator agent can apply within a step.
            predator_min_speed: Minimum speed a predator agent can move at.
            predator_max_speed: Maximum speed a predator agent can move at.
            predator_view_angle: Predator agent local view angle. Should be
                a value from [0,1] representing a fraction of pi radians.
                The view cone of an agent goes from +- of the view angle
                relative to its heading.
            prey_max_rotate: Maximum rotation prey agents can
                turn within a step. Should be a value from [0,1]
                representing a fraction of pi radians.
            prey_max_accelerate: Maximum acceleration/deceleration
                a prey agent can apply within a step.
            prey_min_speed: Minimum speed a prey agent can move at.
            prey_max_speed: Maximum speed a prey agent can move at.
            prey_view_angle: Prey agent local view angle. Should be
                a value from [0,1] representing a fraction of pi radians.
                The view cone of an agent goes from +- of the view angle
                relative to its heading.
            max_steps: Maximum number of environment steps before termination
            viewer: `Viewer` used for rendering. Defaults to `PredatorPreyViewer`.
            generator: Initial state generator. Defaults to `RandomGenerator`.
            reward_fn: Reward function. Defaults to `DistanceRewards`.
        """
        self.num_predators = num_predators
        self.num_prey = num_prey
        self.prey_vision_range = prey_vision_range
        self.predator_vision_range = predator_vision_range
        self.num_vision = num_vision
        self.agent_radius = agent_radius
        self.sparse_rewards = sparse_rewards
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
        self.max_steps = max_steps
        super().__init__()
        self._viewer = viewer or PredatorPreyViewer()
        self._generator = generator or RandomGenerator(num_predators, num_prey)
        self._reward_fn = reward_fn or DistanceRewards(
            predator_vision_range, prey_vision_range, 1.0, 1.0
        )

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
                f" - generator: {self._generator.__class__.__name__}",
                f" - reward-fn: {self._reward_fn.__class__.__name__}",
            ]
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Randomly initialise predator and prey positions and velocities.

        Args:
            key: Random key used to reset the environment.

        Returns:
            state: Agent states.
            timestep: TimeStep with individual agent local environment views.
        """
        state = self._generator(key, self.predator_params, self.prey_params)
        timestep = restart(observation=self._state_to_observation(state))
        return state, timestep

    def step(
        self, state: State, action: Actions
    ) -> Tuple[State, TimeStep[Observation]]:
        """Environment update

        Update agent velocities and consequently their positions,
        them generate new local views and rewards.

        Args:
            state: Agent states.
            action: Arrays of predator and prey individual actions.

        Returns:
            state: Updated agent positions and velocities.
            timestep: Transition timestep with individual agent local observations.
        """
        predators = update_state(
            state.key, self.predator_params, state.predators, action.predators
        )
        prey = update_state(state.key, self.prey_params, state.prey, action.prey)

        state = State(
            predators=predators, prey=prey, key=state.key, step=state.step + 1
        )
        rewards = self._reward_fn(state)
        observation = self._state_to_observation(state)
        timestep = jax.lax.cond(
            state.step >= self.max_steps,
            termination,
            transition,
            rewards,
            observation,
        )
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

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec.

        Local predator and prey agent views representing
        the distance to closest neighbours in the environment.

        Returns:
            observation_spec: Predator-prey observation spec
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

        Arrays of individual agent actions. Each agents action is
        an array representing [rotation, acceleration] in the range
        [-1, 1].

        Returns:
            action_spec: Predator-prey action spec
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
        """Returns the reward spec.

        Arrays of individual rewards for both predator and
        prey types.

        Returns:
            reward_spec: Predator-prey reward spec
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

    def render(self, state: State) -> None:
        """Render a frame of the environment for a given state using matplotlib.

        Args:
            state: State object containing the current dynamics of the environment.
        """
        self._viewer.render(state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> FuncAnimation:
        """Create an animation from a sequence of environment states.

        Args:
            states: sequence of environment states corresponding to consecutive
                timesteps.
            interval: delay between frames in milliseconds.
            save_path: the path where the animation file should be saved. If it
                is None, the plot will not be saved.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """
        return self._viewer.animate(states, interval=interval, save_path=save_path)

    def close(self) -> None:
        """Perform any necessary cleanup."""
        self._viewer.close()
