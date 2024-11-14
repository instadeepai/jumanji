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
from jumanji.environments.swarms.search_and_rescue import utils
from jumanji.environments.swarms.search_and_rescue.dynamics import RandomWalk, TargetDynamics
from jumanji.environments.swarms.search_and_rescue.generator import Generator, RandomGenerator
from jumanji.environments.swarms.search_and_rescue.types import Observation, State, TargetState
from jumanji.environments.swarms.search_and_rescue.viewer import SearchAndRescueViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class SearchAndRescue(Environment):
    """A multi-agent search environment

    Environment modelling a collection of agents collectively searching
    for a set of targets on a 2d environment. Agents are rewarded
    (individually) for coming within a fixed range of a target that has
    not already been detected. Agents visualise their local environment
    (i.e. the location of other agents) via a simple segmented view model.
    The environment consists of a uniform space with wrapped boundaries.

    - observation: `Observation`
        searcher_views: jax array (float) of shape (num_searchers, num_vision)
            individual local views of positions of other searching agents.
        targets_remaining: (float) Number of targets remaining to be found from
            the total scaled to the range [0, 1] (i.e. a value of 1.0 indicates
            all the targets are still to be found).
        time_remaining: (float) Steps remaining to find agents, scaled to the
            range [0,1] (i.e. the value is 0 when time runs out).

    - action: jax array (float) of shape (num_searchers, 2)
        Array of individual agent actions. Each agents actions rotate and
        accelerate/decelerate the agent as [rotation, acceleration] on the range
        [-1, 1]. These values are then scaled to update agent velocities within
        given parameters.

    - reward: jax array (float) of shape (num_searchers,)
        Arrays of individual agent rewards. Rewards are granted when an agent
        comes into contact range with a target that has not yet been found, and
        that agent is within the searchers view cone.

    - state: `State`
        - searchers: `AgentState`
            - pos: jax array (float) of shape (num_searchers, 2) in the range [0, 1].
            - heading: jax array (float) of shape (num_searcher,) in
                the range [0, 2pi].
            - speed: jax array (float) of shape (num_searchers,) in the
                range [min_speed, max_speed].
        - targets: `TargetState`
            - pos: jax array (float) of shape (num_targets, 2) in the range [0, 1].
            - found: jax array (bool) of shape (num_targets,) flag indicating if
                target has been located by an agent.
        - key: jax array (uint32) of shape (2,)
        - step: int representing the current simulation step.


    ```python
    from jumanji.environments import SearchAndRescue
    env = SearchAndRescue(
        searcher_vision_range=0.1,
        target_contact_range=0.01,
        num_vision=40,
        agent_radius0.01,
        searcher_max_rotate=0.1,
        searcher_max_accelerate=0.01,
        searcher_min_speed=0.01,
        searcher_max_speed=0.05,
        searcher_view_angle=0.5,
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
        searcher_vision_range: float,
        target_contact_range: float,
        num_vision: int,
        agent_radius: float,
        searcher_max_rotate: float,
        searcher_max_accelerate: float,
        searcher_min_speed: float,
        searcher_max_speed: float,
        searcher_view_angle: float,
        max_steps: int = 400,
        viewer: Optional[Viewer[State]] = None,
        target_dynamics: Optional[TargetDynamics] = None,
        generator: Optional[Generator] = None,
    ) -> None:
        """Instantiates a `SearchAndRescue` environment

        Note:
            The environment is square with dimensions
            `[1.0, 1.0]` so parameters should be scaled
            appropriately. Also note that performance is
            dependent on agent vision and interaction ranges,
            where larger values can lead to large number of
            agent interactions.

        Args:
            searcher_vision_range: Search agent vision range.
            target_contact_range: Range at which a searcher can 'find' a target.
            num_vision: Number of cells/subdivisions in agent
                view models. Larger numbers provide a more accurate
                view, at the cost of the environment, at the cost
                of performance and memory usage.
            agent_radius: Radius of individual agents. This
                effects how large they appear to other agents.
            searcher_max_rotate: Maximum rotation searcher agents can
                turn within a step. Should be a value from [0,1]
                representing a fraction of pi radians.
            searcher_max_accelerate: Maximum acceleration/deceleration
                a searcher agent can apply within a step.
            searcher_min_speed: Minimum speed a searcher agent can move at.
            searcher_max_speed: Maximum speed a searcher agent can move at.
            searcher_view_angle: Predator agent local view angle. Should be
                a value from [0,1] representing a fraction of pi radians.
                The view cone of an agent goes from +- of the view angle
                relative to its heading.
            max_steps: Maximum number of environment steps allowed for search.
            viewer: `Viewer` used for rendering. Defaults to `SearchAndRescueViewer`.
                target_dynamics:
            target_dynamics: Target object dynamics model, implemented as a
                `TargetDynamics` interface. Defaults to `RandomWalk`.
            generator: Initial state `Generator` instance. Defaults to `RandomGenerator`
                with 20 targets and 10 searchers.
        """
        self.searcher_vision_range = searcher_vision_range
        self.target_contact_range = target_contact_range
        self.num_vision = num_vision
        self.agent_radius = agent_radius
        self.searcher_params = AgentParams(
            max_rotate=searcher_max_rotate,
            max_accelerate=searcher_max_accelerate,
            min_speed=searcher_min_speed,
            max_speed=searcher_max_speed,
            view_angle=searcher_view_angle,
        )
        self.max_steps = max_steps
        self._viewer = viewer or SearchAndRescueViewer()
        self._target_dynamics = target_dynamics or RandomWalk(0.01)
        self.generator = generator or RandomGenerator(num_targets=20, num_searchers=10)
        super().__init__()

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Search & rescue multi-agent environment:",
                f" - num searchers: {self.generator.num_searchers}",
                f" - num targets: {self.generator.num_targets}",
                f" - search vision range: {self.searcher_vision_range}",
                f" - target contact range: {self.target_contact_range}",
                f" - num vision: {self.num_vision}",
                f" - agent radius: {self.agent_radius}",
                f" - max steps: {self.max_steps},"
                f" - target dynamics: {self._target_dynamics.__class__.__name__}",
                f" - generator: {self.generator.__class__.__name__}",
            ]
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Initialise searcher positions and velocities, and target positions.

        Args:
            key: Random key used to reset the environment.

        Returns:
            state: Initial environment state.
            timestep: TimeStep with individual search agent views.
        """
        state = self.generator(key, self.searcher_params)
        timestep = restart(observation=self._state_to_observation(state))
        return state, timestep

    def step(self, state: State, actions: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        """Environment update.

        Update searcher velocities and consequently their positions,
        mark found targets, and generate rewards and local observations.

        Args:
            state: Environment state.
            actions: Arrays of searcher steering actions.

        Returns:
            state: Updated searcher and target positions and velocities.
            timestep: Transition timestep with individual agent local observations.
        """
        searchers = update_state(state.key, self.searcher_params, state.searchers, actions)
        key, target_key = jax.random.split(state.key, num=2)
        # Ensure target positions are wrapped
        target_pos = self._target_dynamics(target_key, state.targets.pos) % 1.0
        # Grant searchers rewards if in range and not already detected
        # spatial maps the has_found_target function over all pair of targets and
        # searchers within range of each other and sums rewards per agent.
        rewards = spatial(
            utils.has_found_target,
            reduction=jnp.add,
            default=0.0,
            i_range=self.target_contact_range,
        )(
            key,
            self.searcher_params.view_angle,
            searchers,
            state.targets,
            pos=searchers.pos,
            pos_b=target_pos,
        )
        # Mark targets as found if with contact range and view angle of a searcher
        # spatial maps the has_been_found function over all pair of targets and
        # searchers within range of each other
        targets_found = spatial(
            utils.has_been_found,
            reduction=jnp.logical_or,
            default=False,
            i_range=self.target_contact_range,
        )(
            key,
            self.searcher_params.view_angle,
            state.targets.pos,
            searchers,
            pos=target_pos,
            pos_b=searchers.pos,
        )
        # Targets need to remain found if they already have been
        targets_found = jnp.logical_or(targets_found, state.targets.found)
        state = State(
            searchers=searchers,
            targets=TargetState(pos=target_pos, found=targets_found),
            key=key,
            step=state.step + 1,
        )
        observation = self._state_to_observation(state)
        timestep = jax.lax.cond(
            state.step >= self.max_steps | jnp.all(targets_found),
            termination,
            transition,
            rewards,
            observation,
        )
        return state, timestep

    def _state_to_observation(self, state: State) -> Observation:
        searcher_views = spatial(
            view,
            reduction=jnp.minimum,
            default=jnp.ones((self.num_vision,)),
            include_self=False,
            i_range=self.searcher_vision_range,
        )(
            state.key,
            (self.searcher_params.view_angle, self.agent_radius),
            state.searchers,
            state.searchers,
            pos=state.searchers.pos,
            n_view=self.num_vision,
            i_range=self.searcher_vision_range,
        )

        return Observation(
            searcher_views=searcher_views,
            targets_remaining=1.0 - jnp.sum(state.targets.found) / self.generator.num_targets,
            time_remaining=1.0 - state.step / (self.max_steps + 1),
        )

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec.

        Local searcher agent views representing
        the distance to the closest neighbouring agents in the
        environment.

        Returns:
            observation_spec: Search-and-rescue observation spec
        """
        searcher_views = specs.BoundedArray(
            shape=(self.generator.num_searchers, self.num_vision),
            minimum=0.0,
            maximum=1.0,
            dtype=float,
            name="searcher_views",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            searcher_views=searcher_views,
            targets_remaining=specs.BoundedArray(
                shape=(), minimum=0.0, maximum=1.0, name="targets_remaining", dtype=float
            ),
            time_remaining=specs.BoundedArray(
                shape=(), minimum=0.0, maximum=1.0, name="time_remaining", dtype=float
            ),
        )

    @cached_property
    def action_spec(self) -> specs.BoundedArray:
        """Returns the action spec.

        2d array of individual agent actions. Each agents action is
        an array representing [rotation, acceleration] in the range
        [-1, 1].

        Returns:
            action_spec: Action array spec
        """
        return specs.BoundedArray(
            shape=(self.generator.num_searchers, 2),
            minimum=-1.0,
            maximum=1.0,
            dtype=float,
        )

    @cached_property
    def reward_spec(self) -> specs.BoundedArray:
        """Returns the reward spec.

        Array of individual rewards for each agent.

        Returns:
            reward_spec: Reward array spec.
        """
        return specs.BoundedArray(
            shape=(self.generator.num_searchers,),
            minimum=0.0,
            maximum=float(self.generator.num_targets),
            dtype=float,
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
