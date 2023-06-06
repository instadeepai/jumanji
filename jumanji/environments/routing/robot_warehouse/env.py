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
from typing import List, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib
from numpy.typing import NDArray

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.robot_warehouse import utils
from jumanji.environments.routing.robot_warehouse.constants import _SHELVES
from jumanji.environments.routing.robot_warehouse.generator import (
    Generator,
    RandomGenerator,
)
from jumanji.environments.routing.robot_warehouse.types import (
    Action,
    Agent,
    Direction,
    Observation,
    Shelf,
    State,
)
from jumanji.environments.routing.robot_warehouse.utils_agent import (
    set_new_direction_after_turn,
    set_new_position_after_forward,
)
from jumanji.environments.routing.robot_warehouse.utils_shelf import update_shelf
from jumanji.environments.routing.robot_warehouse.viewer import RobotWarehouseViewer
from jumanji.tree_utils import tree_slice
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class RobotWarehouse(Environment[State]):
    """A JAX implementation of the 'Robotic warehouse' environment:
    https://github.com/semitable/robotic-warehouse
    which is described in the paper [1].

    Creates a grid world where multiple agents (robots)
    are supposed to collect shelves, bring them to a goal
    and then return them.

    Below is an example warehouse floor grid:
    the grid layout is instantiated using three arguments

    - shelf_rows: number of vertical shelf clusters
    - shelf_columns: odd number of horizontal shelf clusters
    - column_height: height of each cluster

    A cluster is a set of grouped shelves (two cells wide) represented
    below as

                        XX
    Shelf cluster ->    XX    (this cluster is of height 3)
                        XX

    Grid Layout:

                   shelf columns (here set to 3, i.e.
                     v  v  v      shelf_columns=3, must be an odd number)
                    ----------
                 >  -XX-XX-XX-        ^
    Shelf Row 1 ->  -XX-XX-XX-  Column Height (here set to 3, i.e.
                 >  -XX-XX-XX-        v        column_height=3)
                    ----------
                    -XX----XX-   <
                    -XX----XX-   <- Shelf Row 2 (here set to 2, i.e.
                    -XX----XX-   <              shelf_rows=2)
                    ----------
                    ----GG----

    - G: is the goal positions where agents are rewarded if
    they successfully deliver a requested shelf (i.e toggle the load action
    inside the goal position while carrying a requested shelf).

    The final grid size will be
    - height: (column_height + 1) * shelf_rows + 2
    - width: (2 + 1) * shelf_columns + 1

    The bottom-middle column is removed to allow for
    agents to queue in front of the goal positions

    - action: jax array (int) of shape (num_agents,) containing the action for each agent.
        (0: noop, 1: forward, 2: left, 3: right, 4: toggle_load)

    - reward: jax array (int) of shape (), global reward shared by all agents, +1
        for every successful delivery of a requested shelf to the goal position.

    - episode termination:
        - The number of steps is greater than the limit.
        - Any agent selects an action which causes two agents to collide.

    - state: State
        - grid: an array representing the warehouse floor as a 2D grid with two separate channels
            one for the agents, and one for the shelves
        - agents: a pytree of Agent type with per agent leaves: [position, direction, is_carrying]
        - shelves: a pytree of Shelf type with per shelf leaves: [position, is_requested]
        - request_queue: the queue of requested shelves (by ID).
        - step_count: an integer representing the current step of the episode.
        - action_mask: an array of shape (num_agents, 5) containing the valid actions
            for each agent.
        - key: a pseudorandom number generator key.

    [1] Papoudakis et al., Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms
        in Cooperative Tasks (2021)

    ```python
    from jumanji.environments import RobotWarehouse
    env = RobotWarehouse()
    key = jax.random.PRNGKey(0)
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
        time_limit: int = 500,
        viewer: Optional[Viewer[State]] = None,
    ):
        """Instantiates an `RobotWarehouse` environment.

        Args:
            generator: callable to instantiate environment instances.
                Defaults to `RandomGenerator` with parameters:
                `shelf_rows = 2`,
                `shelf_columns = 3`,
                `column_height = 8`,
                `num_agents = 4`,
                `sensor_range = 1`,
                `request_queue_size = 8`.
            time_limit: the maximum step limit allowed within the environment.
                Defaults to 500.
            viewer: viewer to render the environment. Defaults to `RobotWarehouseViewer`.
        """

        # default generator is: robot_warehouse-tiny-4ag-easy (in original implementation)
        self._generator = generator or RandomGenerator(
            column_height=8,
            shelf_rows=2,
            shelf_columns=3,
            num_agents=4,
            sensor_range=1,
            request_queue_size=8,
        )

        self.goals: List[Tuple[int, int]] = []
        self.grid_size = self._generator.grid_size
        self.request_queue_size = self._generator.request_queue_size

        self.num_agents = self._generator.num_agents
        self.sensor_range = self._generator.sensor_range
        self.highways = self._generator.highways
        self.shelf_ids = self._generator.shelf_ids
        self.not_in_queue_size = self._generator.not_in_queue_size

        self.agent_ids = jnp.arange(self.num_agents)
        self.directions = jnp.array([d.value for d in Direction])
        self.num_obs_features = utils.calculate_num_observation_features(
            self.sensor_range
        )
        self.goals = self._generator.goals
        self.time_limit = time_limit

        # create viewer for rendering environment
        self._viewer = viewer or RobotWarehouseViewer(
            self.grid_size, self.goals, "RobotWarehouse"
        )

    def __repr__(self) -> str:
        return (
            f"RobotWarehouse(\n"
            f"\tgrid_width={self.grid_size[1]!r},\n"
            f"\tgrid_height={self.grid_size[0]!r},\n"
            f"\tnum_agents={self.num_agents!r}, \n"
            ")"
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment.

        Args:
            key: random key used to reset the environment since it is stochastic.

        Returns:
            state: State object corresponding to the new state of the environment.
            timestep: TimeStep object corresponding the first timestep returned by the environment.
        """
        # create environment state
        state = self._generator(key)

        # collect first observations and create timestep
        agents_view = self._make_observations(state.grid, state.agents, state.shelves)
        observation = Observation(
            agents_view=agents_view,
            action_mask=state.action_mask,
            step_count=state.step_count,
        )
        timestep = restart(observation=observation)
        return state, timestep

    def step(
        self,
        state: State,
        action: chex.Array,
    ) -> Tuple[State, TimeStep[Observation]]:
        """Perform an environment step.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the action to take.
                - 0 no op
                - 1 move forward
                - 2 turn left
                - 3 turn right
                - 4 toggle load

        Returns:
            state: State object corresponding to the next state of the environment.
            timestep: TimeStep object corresponding the timestep returned by the environment.
        """

        # unpack state
        key = state.key
        grid = state.grid
        agents = state.agents
        shelves = state.shelves
        request_queue = state.request_queue

        # check for invalid action -> turn into noops
        actions = utils.get_valid_actions(action, state.action_mask)

        # check for agent collisions
        collisions = jax.vmap(functools.partial(utils.is_collision, grid))(
            agents, actions
        )
        collision = jnp.any(collisions)

        # update agents, shelves and grid
        def update_state_scan(
            carry_info: Tuple[chex.Array, chex.Array, chex.Array, int], action: int
        ) -> Tuple[Tuple[chex.Array, chex.Array, chex.Array, int], None]:
            grid, agents, shelves, agent_id = carry_info
            grid, agents, shelves = self._update_state(
                grid, agents, shelves, action, agent_id
            )
            return (grid, agents, shelves, agent_id + 1), None

        (grid, agents, shelves, _), _ = jax.lax.scan(
            update_state_scan, (grid, agents, shelves, 0), actions
        )

        # compute shared reward for all agents and update request queue
        # if a requested shelf has been successfully delivered to the goal
        reward = jnp.array(0, dtype=jnp.float32)

        def update_reward_and_request_queue_scan(
            carry_info: Tuple[
                chex.PRNGKey, chex.Array, chex.Array, chex.Array, chex.Array
            ],
            goal: chex.Array,
        ) -> Tuple[
            Tuple[chex.PRNGKey, chex.Array, chex.Array, chex.Array, chex.Array], None
        ]:
            key, reward, request_queue, grid, shelves = carry_info
            (
                key,
                reward,
                request_queue,
                shelves,
            ) = self._update_reward_and_request_queue(
                key, reward, request_queue, grid, shelves, goal
            )
            carry_info = (key, reward, request_queue, grid, shelves)
            return carry_info, None

        update_info, _ = jax.lax.scan(
            update_reward_and_request_queue_scan,
            (key, reward, request_queue, grid, shelves),
            self.goals,
        )
        key, reward, request_queue, grid, shelves = update_info

        # construct timestep and check environment termination
        steps = state.step_count + 1
        horizon_reached = steps >= self.time_limit
        done = collision | horizon_reached

        # compute next observation
        agents_view = self._make_observations(grid, agents, shelves)
        action_mask = utils.compute_action_mask(grid, agents)
        next_observation = Observation(
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=steps,
        )

        timestep = jax.lax.cond(
            done,
            termination,
            transition,
            reward,
            next_observation,
        )
        next_state = State(
            grid=grid,
            agents=agents,
            shelves=shelves,
            request_queue=request_queue,
            step_count=steps,
            action_mask=action_mask,
            key=key,
        )
        return next_state, timestep

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specification of the observation of the `RobotWarehouse` environment.
        Returns:
            Spec for the `Observation`, consisting of the fields:
                - agents_view: Array (int32) of shape (num_agents, num_obs_features).
                - action_mask: BoundedArray (bool) of shape (num_agent, 5).
                - step_count: BoundedArray (int32) of shape ().
        """
        agents_view = specs.Array(
            (self.num_agents, self.num_obs_features), jnp.int32, "agents_view"
        )
        action_mask = specs.BoundedArray(
            (self.num_agents, 5), bool, False, True, "action_mask"
        )
        step_count = specs.BoundedArray((), jnp.int32, 0, self.time_limit, "step_count")
        return specs.Spec(
            Observation,
            "ObservationSpec",
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec. 5 actions: [0,1,2,3,4] -> [No Op, Forward, Left, Right, Toggle_load].
        Since this is a multi-agent environment, the environment expects an array of actions.
        This array is of shape (num_agents,).
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.array([len(Action)] * self.num_agents, jnp.int32),
            name="action",
        )

    def _make_observations(
        self,
        grid: chex.Array,
        agents: Agent,
        shelves: Shelf,
    ) -> chex.Array:
        """Create an observation for each agent based on its view of other
        agents and shelves

        Args:
            grid: the warehouse floor grid array.
            agents: a pytree of Agent type containing agents information.
            shelves: a pytree of Shelf type containing shelves information.

        Returns:
            an array containing agents observations.
        """
        return jax.vmap(
            functools.partial(
                utils.make_agent_observation,
                grid,
                agents,
                shelves,
                self.sensor_range,
                self.num_obs_features,
                self.highways,
            )
        )(self.agent_ids)

    def _update_state(
        self,
        grid: chex.Array,
        agents: chex.Array,
        shelves: chex.Array,
        action: int,
        agent_id: int,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Update the state of the environment after an action is performed.

        Args:
            grid: the warehouse floor grid array.
            agents: a pytree of Agent type containing agents information.
            shelves: a pytree of Shelf type containing shelves information.
            action: the action performed by the agent.
            agent_id: the id of the agent performing the action.
        Returns:
            the updated warehouse floor grid array, agents and shelves.
        """
        agent = tree_slice(agents, agent_id)
        is_highway = self.highways[agent.position.x, agent.position.y]
        grid, agents, shelves = jax.lax.cond(
            jnp.equal(action, Action.FORWARD.value),
            set_new_position_after_forward,
            set_new_direction_after_turn,
            grid,
            agents,
            shelves,
            action,
            agent_id,
            is_highway,
        )

        return grid, agents, shelves

    def _update_reward_and_request_queue(
        self,
        key: chex.PRNGKey,
        reward: chex.Array,
        request_queue: chex.Array,
        grid: chex.Array,
        shelves: chex.Array,
        goal: chex.Array,
    ) -> Tuple[chex.PRNGKey, int, chex.Array, chex.Array]:
        """Check if a shelf has been delivered successfully to a goal state,
        if so reward the agents and update the request queue: removing the ID
        of the delivered shelf and replacing it with a new shelf ID.

        Args:
            key: a pseudorandom number generator key.
            reward: the array of shared reward for each agent.
            request_queue: the queue of requested shelves.
            grid: the warehouse floor grid array.
            shelves: a pytree of Shelf type containing shelves information.
            goal: array of goal positions.
        Returns:
            a random key, updated reward, request queue and shelves.
        """
        x, y = goal
        shelf_id = grid[_SHELVES, x, y]

        def reward_and_update_request_queue_if_shelf_in_goal(
            key: chex.PRNGKey,
            reward: jnp.int32,
            request_queue: chex.Array,
            shelves: chex.Array,
            shelf_id: int,
        ) -> Tuple[chex.PRNGKey, int, chex.Array, chex.Array]:
            "Reward the agents and update the request queue."

            # remove from queue and replace it
            key, request_key = jax.random.split(key)

            not_in_queue = jnp.setdiff1d(
                self.shelf_ids,
                request_queue,
                size=self.not_in_queue_size,
            )
            new_request_id = jax.random.choice(
                request_key,
                not_in_queue,
                replace=False,
            )
            replace_index = jnp.argwhere(jnp.equal(request_queue, shelf_id - 1), size=1)
            request_queue = request_queue.at[replace_index].set(new_request_id)

            # also reward the agents
            reward += 1.0

            # update requested shelf
            shelves = update_shelf(shelves, shelf_id - 1, "is_requested", 0)
            shelves = update_shelf(shelves, new_request_id, "is_requested", 1)
            return key, reward, request_queue, shelves

        # check if shelf is at goal position and in request queue
        cond = (shelf_id != 0) & jnp.isin(shelf_id, request_queue + 1)

        key, reward, request_queue, shelves = jax.lax.cond(
            cond,
            reward_and_update_request_queue_if_shelf_in_goal,
            lambda k, r, rq, g, _: (k, r, rq, g),
            key,
            reward,
            request_queue,
            shelves,
            shelf_id,
        )
        return key, reward, request_queue, shelves

    def render(self, state: State) -> Optional[NDArray]:
        """Renders the current state of the RobotWarehouse environment.

        Args:
            state: is the current environment state to be rendered.
            save_path: the path where the image should be saved. If it is None, the plot
            will not be stored.
        """
        return self._viewer.render(state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Creates an animation from a sequence of RobotWarehouse states.

        Args:
            states: sequence of `State` corresponding to subsequent timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None,
                the plot will not be saved.

        Returns:
            Animation object that can be saved as a GIF, MP4, or rendered with HTML.
        """
        return self._viewer.animate(
            states=states, interval=interval, save_path=save_path
        )

    def close(self) -> None:
        """Perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        self._viewer.close()
