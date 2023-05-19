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

from typing import Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib
from chex import PRNGKey
from numpy.typing import NDArray

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.tsp.generator import Generator, UniformGenerator
from jumanji.environments.routing.tsp.reward import DenseReward, RewardFn
from jumanji.environments.routing.tsp.types import Observation, State
from jumanji.environments.routing.tsp.viewer import TSPViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class TSP(Environment[State]):
    """Traveling Salesman Problem (TSP) environment as described in [1].

    - observation: Observation
        - coordinates: jax array (float) of shape (num_cities, 2)
            the coordinates of each city.
        - position: jax array (int32)  of shape ()
            the index corresponding to the last visited city.
        - trajectory: jax array (int32) of shape (num_cities,)
            array of city indices defining the route (-1 --> not filled yet).
        - action_mask: jax array (bool) of shape (num_cities,)
            binary mask (False/True <--> illegal/legal <--> cannot be visited/can be visited).

    - action: jax array (int32) of shape ()
        [0, ..., num_cities - 1] -> city to visit.

    - reward: jax array (float) of shape (), could be either:
        - dense: the negative distance between the current city and the chosen next city to go to.
            It is 0 for the first chosen city, and for the last city, it also includes the distance
            to the initial city to complete the tour.
        - sparse: the negative tour length at the end of the episode. The tour length is defined
            as the sum of the distances between consecutive cities. It is computed by starting at
            the first city and ending there, after visiting all the cities.
        In both cases, the reward is a large negative penalty of `-num_cities * sqrt(2)` if
        the action is invalid, i.e. a previously selected city is selected again.

    - episode termination:
        - if no action can be performed, i.e. all cities have been visited.
        - if an invalid action is taken, i.e. an already visited city is chosen.

    - state: `State`
        - coordinates: jax array (float) of shape (num_cities, 2)
            the coordinates of each city.
        - position: int32
            the identifier (index) of the last visited city.
        - visited_mask: jax array (bool) of shape (num_cities,)
            binary mask (False/True <--> not visited/visited).
        - trajectory: jax array (int32) of shape (num_cities,)
            the identifiers of the cities that have been visited (-1 means that no city has been
            visited yet at that time in the sequence).
        - num_visited: int32
            number of cities that have been visited.

    [1] Kwon Y., Choo J., Kim B., Yoon I., Min S., Gwon Y. (2020). "POMO: Policy Optimization
        with Multiple Optima for Reinforcement Learning".

    ```python
    from jumanji.environments import TSP
    env = TSP()
    key = jax.random.key(0)
    state, timestep = jax.jit(env.reset)(key)
    env.render(state)
    action = env.action_spec().generate_value()
    state, timestep = jax.jit(env.step)(state, action)
    env.render(state)
    ```
    """

    def __init__(
        self,
        generator: Optional[Generator] = None,
        reward_fn: Optional[RewardFn] = None,
        viewer: Optional[Viewer[State]] = None,
    ):
        """Instantiates a `TSP` environment.

        Args:
            generator: `Generator` whose `__call__` instantiates an environment instance.
                The default option is 'UniformGenerator' which randomly generates
                TSP instances with 20 cities sampled from a uniform distribution.
            reward_fn: RewardFn whose `__call__` method computes the reward of an environment
                transition. The function must compute the reward based on the current state,
                the chosen action and the next state.
                Implemented options are [`DenseReward`, `SparseReward`]. Defaults to `DenseReward`.
            viewer: `Viewer` used for rendering. Defaults to `TSPViewer` with "human" render mode.
        """

        self.generator = generator or UniformGenerator(
            num_cities=20,
        )
        self.num_cities = self.generator.num_cities
        self.reward_fn = reward_fn or DenseReward()
        self._viewer = viewer or TSPViewer(name="TSP", render_mode="human")

    def __repr__(self) -> str:
        return f"TSP environment with {self.num_cities} cities."

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment.

        Args:
            key: used to randomly generate the coordinates.

        Returns:
            state: State object corresponding to the new state of the environment.
            timestep: TimeStep object corresponding to the first timestep returned
                by the environment.
        """
        state = self.generator(key)
        timestep = restart(observation=self._state_to_observation(state))
        return state, timestep

    def step(
        self, state: State, action: chex.Numeric
    ) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: `State` object containing the dynamics of the environment.
            action: `Array` containing the index of the next position to visit.

        Returns:
            state: the next state of the environment.
            timestep: the timestep to be observed.
        """
        is_valid = ~state.visited_mask[action]
        next_state = jax.lax.cond(
            is_valid,
            self._update_state,
            lambda *_: state,
            state,
            action,
        )

        reward = self.reward_fn(state, action, next_state, is_valid)
        observation = self._state_to_observation(next_state)

        # Terminate if all cities have been visited or the action is invalid
        is_done = (next_state.num_visited == self.num_cities) | ~is_valid
        timestep = jax.lax.cond(
            is_done,
            termination,
            transition,
            reward,
            observation,
        )
        return next_state, timestep

    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec.

        Returns:
            Spec for the `Observation` whose fields are:
            - coordinates: BoundedArray (float) of shape (num_cities,).
            - position: DiscreteArray (num_values = num_cities) of shape ().
            - trajectory: BoundedArray (int32) of shape (num_cities,).
            - action_mask: BoundedArray (bool) of shape (num_cities,).
        """
        coordinates = specs.BoundedArray(
            shape=(self.num_cities, 2),
            minimum=0.0,
            maximum=1.0,
            dtype=float,
            name="coordinates",
        )
        position = specs.DiscreteArray(
            self.num_cities, dtype=jnp.int32, name="position"
        )
        trajectory = specs.BoundedArray(
            shape=(self.num_cities,),
            dtype=jnp.int32,
            minimum=-1,
            maximum=self.num_cities - 1,
            name="trajectory",
        )
        action_mask = specs.BoundedArray(
            shape=(self.num_cities,),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            coordinates=coordinates,
            position=position,
            trajectory=trajectory,
            action_mask=action_mask,
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec.

        Returns:
            action_spec: a `specs.DiscreteArray` spec.
        """
        return specs.DiscreteArray(self.num_cities, name="action")

    def render(self, state: State) -> Optional[NDArray]:
        """Render the given state of the environment. This rendering shows the layout of the cities
        and the tour so far.

        Args:
            state: current environment state.

        Returns:
            rgb_array: the RGB image of the state as an array.
        """
        return self._viewer.render(state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Creates an animated gif of the `TSP` environment based on the sequence of states.

        Args:
            states: sequence of environment states corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            animation.FuncAnimation: the animation object that was created.
        """
        return self._viewer.animate(states, interval, save_path)

    def close(self) -> None:
        """Perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        self._viewer.close()

    def _update_state(self, state: State, action: chex.Numeric) -> State:
        """
        Updates the state of the environment.

        Args:
            state: State object containing the dynamics of the environment.
            action: int32, index of the next position to visit.

        Returns:
            state: State object corresponding to the new state of the environment.
        """
        return State(
            coordinates=state.coordinates,
            position=action,
            visited_mask=state.visited_mask.at[action].set(True),
            trajectory=state.trajectory.at[state.num_visited].set(action),
            num_visited=state.num_visited + 1,
            key=state.key,
        )

    def _state_to_observation(self, state: State) -> Observation:
        """Converts a state into an observation.

        Args:
            state: `State` object containing the dynamics of the environment.

        Returns:
            observation: `Observation` object containing the observation of the environment.
        """
        return Observation(
            coordinates=state.coordinates,
            position=state.position,
            trajectory=state.trajectory,
            action_mask=~state.visited_mask,
        )
