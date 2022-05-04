import functools
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from dm_env import specs
from jax import random

import jumanji.jax.pcb_grid.env_viewer as viewer
from jumanji.jax.env import JaxEnv
from jumanji.jax.pcb_grid.constants import EMPTY, HEAD, NOOP, SOURCE, TARGET
from jumanji.jax.pcb_grid.types import Position, State
from jumanji.jax.types import Extra, TimeStep, restart, termination, transition


class PcbGridEnv(JaxEnv[State]):
    """
    A JAX implementation of the 'PCB Grid' environment. Currently hard mode is not supported.

    - observation: jax array (int) of shape (num_agents, rows, cols):
        - each 2d array (row, col) along axis 0 is the agent's local observation.
        - agents have ids from 0 to (num_agents - 1)

    - reward: jax array (float) of shape (num_agents,)
        - each agents reward is given separately

    - episode termination: if a wire traps itself and can't move
        or the horizon is reached, or the wire connects to its target it is considered done.
        Once all wires are done, the episode will terminate.
        - timestep discounts are in the shape (num_agents,)

    - state: State:
        - key : jax array (uint) of shape (2,) used to randomly spawn agents and targets.
        - grid : jax array (int) of shape (rows, cols) which corresponds to agent 0's observation.
            This is all that is needed to represent state and is used internally for most functions.
        - step : int, step number in episode.
    """

    metadata = {"render.modes": ["human", "fast"]}
    VIEWER_WIDTH = 1000
    VIEWER_HEIGHT = 1000

    def __init__(
        self,
        rows: int = 12,
        cols: int = 12,
        num_agents: int = 1,
        difficulty: str = "easy",
        reward_per_timestep: float = -0.03,
        reward_for_connection: float = 0.1,
        reward_for_blocked: float = -0.1,
        step_limit: int = 50,
        reward_for_terminal_step: float = -0.1,
        renderer: Optional[viewer.PcbGridViewer] = None,
    ):
        """Create the PCB Grid Environment.

        Args:
            rows : number of rows in the grid.
            cols : number of columns in the grid.
            num_agents : number of separate agents/wires and separate targets in the grid.
            difficulty : there are two difficulties - "easy" is normal random spawning of
                wires and targets - "hard" ensures wires have straight line intersections
                with other agents making it more difficult to connect and requiring cooperation.
                Currently hard mode is not supported.
            reward_per_timestep : the reward given to an agent for every
                timestep not being connected.
            reward_for_connection : the reward given to an agent for
                connecting to its target.
            reward_for_blocked : the reward given to an agent
                for blocking itself.
            step_limit : the number of steps allowed before an episode terminates.
            reward_for_terminal_step : the reward given if step_limit is reached.
            renderer: an optional PcbGridViewer instance to render the environment, if left as None
                a default viewer is created when render is called.
        """

        self.rows = rows
        self.cols = cols
        self.num_agents = num_agents
        self.difficulty = difficulty.upper()
        if self.difficulty != "EASY":
            raise NotImplementedError(
                f"'easy' is the only implemented mode, but got difficulty: {difficulty}."
            )

        self.obs_ints = 2 + 3 * num_agents

        self._reward_time_step = jnp.array(reward_per_timestep, float)
        self._reward_connected = jnp.array(reward_for_connection, float)
        self._reward_blocked = jnp.array(reward_for_blocked, float)

        self._step_limit = step_limit
        self._reward_for_terminal_step = jnp.array(reward_for_terminal_step, float)

        if renderer:
            assert isinstance(
                renderer, viewer.PcbGridViewer
            ), f"Expected a renderer of type 'PcbGridViewer', got {renderer} of type {type(renderer)}."
        self.viewer = renderer

    def observation_spec(self) -> specs.Array:
        """Returns the observation spec.

        Returns:
            observation_spec: dm_env.specs object
        """
        return specs.BoundedArray(
            shape=(self.num_agents, self.rows, self.cols),
            dtype=jnp.int_,
            name="observation",
            minimum=0,
            maximum=self.obs_ints,
        )

    def action_spec(self) -> specs.BoundedArray:
        """Returns the action spec. 5 actions: [0,1,2,3,4] -> [No Op, Left, Up, Right, Down].
        Since this is a multi-agent environment, the environment expects an array of actions.
        This array is of shape (num_agents,)

        Returns:
            action_spec: dm_env.specs object
        """
        return specs.BoundedArray((self.num_agents,), jnp.int_, 0, 4)

    def reward_spec(self) -> specs.Array:
        """Returns the reward spec.
        Since this is a multi-agent environment, the environment gives an array of rewards.
        This array is of shape (num_agents,)

        Returns:
            reward_spec: a `dm_env.specs.Array` spec.
        """
        return specs.Array(shape=(self.num_agents,), dtype=jnp.float_, name="reward")

    def discount_spec(self) -> specs.Array:
        """Describes the discount returned by the environment.
            Since this is a multi-agent environment, the environment gives an array of discounts.
            This array is of shape (num_agents,)

        Returns:
            discount_spec: a `dm_env.specs.Array` spec.
        """
        return specs.BoundedArray(
            shape=(self.num_agents,),
            dtype=jnp.float_,
            minimum=0.0,
            maximum=1.0,
            name="discount",
        )

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep[Array], Extra]:
        """Resets the environment.

        Args:
            key: random key used to reset the environment since it is stochastic.

        Returns:
            state: State object corresponding to the new state of the environment,
            timestep: TimeStep object corresponding the first timestep returned by the environment,
            extra: metrics, default to None."""
        grid = jnp.zeros((self.rows, self.cols), int)
        state_key, spawn_key = random.split(key)
        spawn_keys = random.split(spawn_key, self.num_agents)

        def spawn_scan(
            grid_and_agent_id: Tuple[Array, int], key: PRNGKey
        ) -> Tuple[Tuple[Array, int], None]:
            grid, agent_id = grid_and_agent_id
            grid = self._spawn_agent(grid, key, agent_id)
            return (grid, agent_id + 1), None

        (grid, _), _ = jax.lax.scan(spawn_scan, (grid, 0), spawn_keys)
        observations = self._get_obsv(grid)
        timestep = restart(observation=observations, shape=self.num_agents)
        state = State(
            key=state_key,
            grid=grid,
            step=jnp.array(0, int),
            finished_agents=jnp.zeros(self.num_agents, bool),
        )
        return state, timestep, None

    def step(self, state: State, action: Array) -> Tuple[State, TimeStep[Array], Extra]:
        """Perform an environment step.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the actions to take.
                - 0 no op
                - 1 move left
                - 2 move up
                - 3 move right
                - 4 move down

        Returns:
            state: State object corresponding to the next state of the environment,
            timestep: TimeStep object corresponding the timestep returned by the environment,
            extra: metrics, default to None.
        """

        def step_agent_fn(grid: Array, agent_id: int) -> Tuple[Array, None]:
            new_grid = self._step_agent(grid, agent_id, action[agent_id])
            return new_grid, None

        grid, _ = jax.lax.scan(step_agent_fn, state.grid, jnp.arange(self.num_agents))

        observations = self._get_obsv(grid)
        finished_agents = self.get_finished_agents(grid)
        rewards = self._get_rewards(grid) * ~state.finished_agents
        horizon_reached = state.step >= self._step_limit

        rewards = jax.lax.cond(
            horizon_reached,
            lambda r: r + self._reward_for_terminal_step,
            lambda r: r,
            rewards,
        )

        timestep: TimeStep[Array] = jax.lax.cond(
            jnp.all(finished_agents) | horizon_reached,
            lambda _: termination(
                reward=rewards, observation=observations, shape=self.num_agents
            ),
            lambda _: transition(
                reward=rewards,
                observation=observations,
                discount=jnp.logical_not(finished_agents).astype(float),
            ),
            None,
        )

        next_state = State(
            key=state.key,
            grid=grid,
            step=state.step + 1,
            finished_agents=finished_agents,
        )

        return next_state, timestep, None

    def get_action_mask(self, grid: Array, agent_id: int) -> Array:
        """
        Gets an agents action mask.

        Args:
            grid (Array): the environment state grid.
            agent_id: (int): the agent id whose legal actions are being checked.

        Return:
            Array : action mask.
        """

        def _is_free_cell(grid: Array, x: Array, y: Array, target: Array) -> jnp.bool_:
            return (grid[x, y] == EMPTY) | (grid[x, y] == target)

        mask = jnp.zeros(5, int).at[0].set(1)
        pos, target = self._extract_agent_information(grid, agent_id)
        target = grid[target.x, target.y]

        # left
        mask = mask.at[1].set(
            jnp.array(
                (pos.y > 0) & (_is_free_cell(grid, pos.x, pos.y - 1, target)), int
            )
        )
        # up
        mask = mask.at[2].set(
            jnp.array(
                (pos.x > 0) & (_is_free_cell(grid, pos.x - 1, pos.y, target)), int
            )
        )
        # right
        mask = mask.at[3].set(
            jnp.array(
                (pos.y < self.cols - 1)
                & (_is_free_cell(grid, pos.x, pos.y + 1, target)),
                int,
            )
        )
        # down
        mask = mask.at[4].set(
            jnp.array(
                (pos.x < self.rows - 1)
                & (_is_free_cell(grid, pos.x + 1, pos.y, target)),
                int,
            )
        )

        return jax.lax.cond(
            self.is_agent_connected(grid, agent_id),
            lambda _: jnp.array([1, 0, 0, 0, 0], int),
            lambda _: mask,
            None,
        )

    def render(self, grid: Array, mode: str = "human") -> Array:
        """Render a state in pygame.

        Args:
            grid (Array) : the environment state grid to render.
            sleep (float): amount of time to sleep afterwards - this helps to slow
                down each step.

        Return:
            Array of rgb pixel values in the shape width x height x rgb"""
        if self.viewer is None:
            self.viewer = viewer.PcbGridViewer(
                self.num_agents,
                self.rows,
                self.cols,
                self.VIEWER_WIDTH,
                self.VIEWER_HEIGHT,
            )

        return self.viewer.render(grid, mode)

    def _extract_agent_information(
        self, grid: Array, agent_id: int
    ) -> Tuple[Position, Position]:
        """Extract the agent's head and target position in the state grid.

        Args:
            grid (Array): the environment state grid (self.rows, self.cols).
            agent_id (int): the agents id.

        Return:
            Tuple[Position, Position]: the agent's head and target position in the grid."""
        d = agent_id * 3
        head = jnp.where(grid == 4 + d, size=1)
        head = Position(x=head[0][0], y=head[1][0])
        target = jnp.where(grid == 3 + d, size=1)
        target = Position(x=target[0][0], y=target[1][0])
        return head, target

    @staticmethod
    def is_agent_connected(grid: Array, agent_id: int) -> jnp.bool_:
        """Checks to see if an agent is done.

        Args:
            grid (Array): the environment state grid.
            agent_id (int): the agents id.
        Return:
            bool : true if agent is connected otherwise false.
        """
        d = agent_id * 3
        target = 3 + d

        return jnp.logical_not(jnp.isin(target, grid))

    def _get_obsv(self, grid: Array) -> Array:
        """
        Get each agents individual observations and compile
        them together into the state grid representation.

        Args:
            grid (Array): The environment state grid.

        Return:
            Array: each agents local observation stacked on top of
                each other in the shape (num_agents, rows, cols).
        """
        return jax.vmap(functools.partial(self._agent_observation, grid))(
            jnp.arange(self.num_agents, dtype=jnp.int_)
        )

    def _spawn_agent(self, grid: Array, key: PRNGKey, agent_id: int) -> Array:
        """Spawn an agent and update the state grid.

        Args:
            grid (Array): environment state grid.
            key (KeyArray): pseudo random number key.
            id (int): id of the agent being spawned.

        Return:
            Array: the updated environment grid."""
        pos_key, target_key = random.split(key)
        agent_pos = self._random_empty_position(grid, pos_key)
        new_state = grid.at[agent_pos].set(HEAD + 3 * agent_id)
        agent_target = self._random_empty_position(new_state, target_key)

        new_state = new_state.at[agent_target].set(TARGET + 3 * agent_id)
        return new_state

    def _random_empty_position(self, grid: Array, key: PRNGKey) -> Position:
        """Generates a random empty position in the environment grid using the random key.

        Args:
            grid (Array): the environment state grid.
            key (KeyArray): the pseudo random number key.

        Returns:
            Tuple: an empty position.
            KeyArray : the new pseudo random number key."""
        pos_index = random.choice(
            key, jnp.arange(self.rows * self.cols), p=jnp.logical_not(grid).flatten()
        )
        row, col = divmod(pos_index, self.cols)
        pos = Position(x=row, y=col)
        return pos

    def get_finished_agents(self, grid: Array) -> Array:
        """Get the done flags for each agent.

        Args:
            grid (Array) : the environment state grid.

        Return:
            Array : array of boolean flags in the shape (number of agents, )."""

        def done_fun(grid: Array, agent_id: jnp.int_) -> jnp.bool_:
            return (self._is_agent_blocked(self.get_action_mask(grid, agent_id))) | (
                self.is_agent_connected(grid, agent_id)
            )

        return jax.vmap(functools.partial(done_fun, grid))(
            jnp.arange(self.num_agents, dtype=jnp.int_)
        )

    def _get_rewards(self, grid: Array) -> Array:
        """Get the rewards for each agent.

        Args:
            grid (Array) : the environment state grid.

        Return:
            Array : array of rewards in the shape (number of agents, )."""

        def reward_fun(grid: Array, agent_id: jnp.int_) -> jnp.float_:
            return jax.lax.cond(
                self.is_agent_connected(grid, agent_id),
                lambda _: self._reward_connected,
                lambda _: jax.lax.cond(
                    self._is_agent_blocked(self.get_action_mask(grid, agent_id)),
                    lambda _: self._reward_blocked,
                    lambda _: self._reward_time_step,
                    None,
                ),
                None,
            )

        return jax.vmap(functools.partial(reward_fun, grid))(
            jnp.arange(self.num_agents, dtype=jnp.int_)
        )

    def _agent_observation(self, grid: Array, agent_id: int) -> Array:
        """
        Get an observation for an agents perspective, this makes it so an agent always sees
            itself using the values 2, 3, 4.

        Args:
            grid (Array): the environment state grid.
            agent_id (int): the id of the agent whose observation is being requested.

        Return:
            Array: the state in the perspective of the agent."""

        def swap_obs(grid_and_agent_id: Tuple[Array, int]) -> Array:

            grid, agent_id = grid_and_agent_id

            zeros_mask = grid != 0  # to remove all zeros at the env
            ones_mask = grid != 1  # to reset the ones back to zeros
            ones_inds = grid == 1  # to increment the ones back to 1

            # only adding one here because so that we can ignore the leading 0 and 1
            # values that are not agent related. This is equivalent
            # to obs -=2 (to transform the agent values such that
            # they start from 0) then obs += 3 to move the observations
            # to the next agent

            # -= 2 to transform the observations such that the values related to
            # agents lie between 0 and self.obs_ints - 2
            # (0's and 1's, which aren't agent related, can be ignored)
            grid -= 2
            grid -= 3 * (agent_id)
            grid %= self.obs_ints - 2  # max value of state ignoring the 0 and 1
            # making space for the leading 0 and 1 (transforming the state back from
            # the implicit -2 at the begining)
            grid += 2
            grid *= zeros_mask  # adding back the zeros
            grid *= ones_mask  # adding resetting the ones to zero
            grid += ones_inds  # adding back the ones

            return grid

        return jax.lax.cond(agent_id == 0, lambda x: x[0], swap_obs, (grid, agent_id))

    def _is_agent_blocked(self, action_mask: Array) -> jnp.bool_:
        """Checks if an agent is blocked by checking their action mask.

        Args:
            action_mask (Array): an agents action mask.

        Return:
            bool: returns true if the agent is blocked otherwise false."""
        return jnp.all(action_mask[1:] == 0)

    def _step_agent(self, grid: Array, agent_id: int, action: int) -> Array:
        """Take an agent step.

        Args:
            grid (Array): the environment state grid.
            id (int): the agents id.
            action (int): the agents action.

        Return:
            Array: the new updated state."""
        pos, _ = self._extract_agent_information(grid, agent_id)
        position = move(pos, action)
        is_not_valid = jnp.logical_not(self._is_valid(grid, agent_id, position))
        return jax.lax.cond(
            ((action == NOOP) | (is_not_valid)),
            lambda x: x[0],
            lambda x: self._move_agent(x),
            (grid, agent_id, position),
        )

    def _is_valid(
        self,
        grid: Array,
        agent_id: int,
        position: Position,
    ) -> jnp.bool_:
        """Checks to see if an agents move is valid.

        Args:
            grid (Array): the environment state grid.
            agent_id (int): the agents id.
            position (Tuple[Array,Array]): the agents position.

        Return:
            bool: True if the agent moving to position is valid."""
        row, col = position.x, position.y
        return (
            (0 <= row)
            & (row < self.rows)
            & (0 <= col)
            & (col < self.cols)
            & ((grid[position] == EMPTY) | (grid[position] == TARGET + 3 * agent_id))
            & (jnp.logical_not(self.is_agent_connected(grid, agent_id)))
        )

    def _move_agent(self, grid_id_position: Tuple[Array, int, Position]) -> Array:
        """Move an agent.

        Args:
            grid (Array): the environment state grid.
            agent_id (Agent): the id of the agent being moved.
            position (tuple(int, int)): the position the agent is moving to.

        Return:
            Array: the new updated environment state."""
        grid, agent_id, position = grid_id_position
        pos, _ = self._extract_agent_information(grid, agent_id)
        new_grid = grid.at[pos].set(SOURCE + 3 * agent_id)

        new_grid = new_grid.at[position].set(HEAD + 3 * agent_id)

        return new_grid

    def __repr__(self) -> str:
        return f"<PCBGridEnv(rows={self.rows}, cols={self.cols}, agents={self.num_agents})>"


def move(position: Position, action: int) -> Position:
    """Use a position and an action to return a new position.

    Args:
        position (Position): a position representing row and col.
        action (int): the action representing cardinal directions.

    Return:
        Position : the new position after the move.
    """

    move_left = lambda position: (position.x, position.y - 1)
    move_up = lambda position: (position.x - 1, position.y)
    move_right = lambda position: (position.x, position.y + 1)
    move_down = lambda position: (position.x + 1, position.y)

    new_pos_row, new_pos_col = jax.lax.switch(
        action - 1, [move_left, move_up, move_right, move_down], position
    )

    new_pos = Position(x=new_pos_row, y=new_pos_col)

    return new_pos


def intersect(
    position_a: Position,
    position_b: Position,
    position_c: Position,
    position_d: Position,
) -> jnp.bool_:
    """Checks to see if line segments intersect.

    Args:
        position A, B, C, D (Position): each argument is a position.

    Return:
        bool: true if line segments AB and CD intersect."""
    return (
        counter_clockwise(position_a, position_c, position_d)
        != counter_clockwise(position_b, position_c, position_d)
    ) & (
        counter_clockwise(position_a, position_b, position_c)
        != counter_clockwise(position_a, position_b, position_d)
    )


def counter_clockwise(
    position_a: Position,
    position_b: Position,
    position_c: Position,
) -> Array:
    """Determines if points A, B and C are positioned in a counter clockwise order.

    Args:
        position A, B, C, D (Position): each argument is a position.

    Return:
        bool: true if points A, B, C are ordered in a counter clockwise manner.
    """
    return (position_c.y - position_a.y) * (position_b.x - position_a.x) > (
        position_b.y - position_a.y
    ) * (position_c.x - position_a.x)
