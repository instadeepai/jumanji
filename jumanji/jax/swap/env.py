from typing import Tuple

import jax.numpy as jnp
from chex import Array, PRNGKey
from dm_env import specs
from jax import lax, nn, random

from jumanji.jax.env import JaxEnv
from jumanji.jax.swap.types import State
from jumanji.jax.types import Action, Extra, TimeStep, restart, transition, truncation


class Swap(JaxEnv[State]):
    """
    A JAX implementation of the two-color toy problem described in (Flennerhag et al., 2021).
    The episode is reset every fixed number of periods.

    - observation: jax array (float) of shape (3, 2, max(n_rows, n_cols)):
        - agent_pos: a pair of one-hot encodings, for the x and y coordinate respectively.
        - blue_pos: a pair of one-hot encodings for each coordinate, for the x and y coordinate
            respectively.
        - red_pos: a pair of one-hot encodings for each coordinate, for the x and y coordinate
            respectively.
        For a board of size (5, 5), an agent in position (2, 3), a blue target in (1, 2) and a
        red target in (1, 4), the observation, whose shape is (3, 2, 5), is the following:
        observation = [
            [
                [0,0,1,0,0], [0,0,0,1,0],
            ],
            [
                [0,1,0,0,0], [0,0,1,0,0],
            ],
            [
                [0,1,0,0,0], [0,0,0,0,1],
            ],
        ]

    - reward: 1 if correct target is reached, -1 if wrong target is reached, -0.04 otherwise.

    - episode termination: there is no termination in this game. Hence, the episode goes forever.

    - state: State:
        - key: jax array (uint) of shape (2,) used for the random generation of new targets.
        - agent_pos: jax array (int) of shape (2,) representing the coordinates of the agent.
        - blue_pos: jax array (int) of shape (2,) representing the coordinates of the blue target.
        - red_pos: jax array (int) of shape (2,) representing the coordinates of the red target.
        - step_count: int, step number in the game.
    """

    def __init__(
        self,
        n_rows: int = 5,
        n_cols: int = 5,
        swap_period: int = 100_000,
        n_periods: int = 1,
    ):
        """Instantiates a Swap environment.

        Args:
            n_rows: number of rows of the 2D grid.
            n_cols: number of columns of the 2D grid.
            swap_period: time period after which the reward for each object flips.
            n_periods: number of periods (swapping blue and red targets twice) before the reset of
                the environment.

        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.swap_period = swap_period
        self.n_periods = n_periods
        self._reward_no_target = jnp.float32(-0.04)
        self._reward_good_target = jnp.float32(1.0)
        self._reward_bad_target = jnp.float32(-1.0)

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Swap environment:",
                f" - n_rows: {self.n_rows}",
                f" - n_cols: {self.n_cols}",
                f" - swap_period: {self.swap_period}",
                f" - n_periods: {self.n_periods}",
            ]
        )

    def observation_spec(self) -> specs.Array:
        """Returns the observation spec.

        Returns:
            observation_spec: dm_env.specs object
        """
        return specs.Array(
            shape=(3, 2, max(self.n_rows, self.n_cols)),
            dtype=jnp.float32,
            name="observation",
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec. 4 actions: [0,1,2,3] -> [Up, Right, Down, Left].

        Returns:
            action_spec: dm_env.specs object
        """
        return specs.DiscreteArray(4, name="action")

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep[Array], Extra]:
        """Resets the environment.

        Args:
            key: random key used to reset the environment since it is stochastic.

        Returns:
            state: State object corresponding to the new state of the environment,
            timestep: TimeStep object corresponding the first timestep returned by the environment,
            extra: metrics, default to None.
        """
        key, sample_key = random.split(key)
        pos_indices = random.choice(
            sample_key, jnp.arange(self.n_rows * self.n_cols), shape=(3,), replace=False
        )
        agent_pos, blue_pos, red_pos = [
            jnp.array([pos_index // self.n_cols, pos_index % self.n_cols])
            for pos_index in pos_indices
        ]
        state = State(
            key=key,
            agent_pos=agent_pos,
            blue_pos=blue_pos,
            red_pos=red_pos,
            step_count=jnp.int32(0),
        )
        obs = jnp.array(
            [
                nn.one_hot(
                    state.agent_pos, max(self.n_rows, self.n_cols), dtype=jnp.float32
                ),
                nn.one_hot(
                    state.blue_pos, max(self.n_rows, self.n_cols), dtype=jnp.float32
                ),
                nn.one_hot(
                    state.red_pos, max(self.n_rows, self.n_cols), dtype=jnp.float32
                ),
            ],
            dtype=jnp.float32,
        )
        timestep = restart(observation=obs)
        return state, timestep, None

    def step(
        self, state: State, action: Action
    ) -> Tuple[State, TimeStep[Array], Extra]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the action number to take.
                - 0 move up
                - 1 move to the right
                - 2 move down
                - 3 move to the left

        Returns:
            state: State object corresponding to the next state of the environment,
            timestep: TimeStep object corresponding the timestep returned by the environment,
            extra: metrics, default to None.
        """
        state, reward = self._update_state(state, action)
        timestep = self._state_to_timestep(state, reward)
        return state, timestep, None

    def _update_state(self, state: State, action: Action) -> Tuple[State, Array]:
        """Update the environment state by taking an action.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the action number to take.

        Returns:
            state: environment state after taking one step.
            reward: reward from taking the action.

        """
        next_key, sample_key = random.split(state.key)

        pos_min = jnp.array((0, 0), dtype=jnp.int32)
        pos_max = jnp.array((self.n_rows - 1, self.n_cols - 1), dtype=jnp.int32)
        pos_change_branches = [
            lambda row_col: jnp.clip(
                jnp.array((row_col[0] - 1, row_col[1]), int), pos_min, pos_max
            ),
            lambda row_col: jnp.clip(
                jnp.array((row_col[0], row_col[1] + 1), int), pos_min, pos_max
            ),
            lambda row_col: jnp.clip(
                jnp.array((row_col[0] + 1, row_col[1]), int), pos_min, pos_max
            ),
            lambda row_col: jnp.clip(
                jnp.array((row_col[0], row_col[1] - 1), int), pos_min, pos_max
            ),
        ]
        next_agent_pos = lax.switch(action, pos_change_branches, state.agent_pos)

        blue_target_reached = jnp.array_equal(next_agent_pos, state.blue_pos)
        next_blue_pos = lax.select(
            blue_target_reached,
            self._sample_pos(next_agent_pos, state.red_pos, sample_key),
            state.blue_pos,
        )

        red_target_reached = jnp.array_equal(next_agent_pos, state.red_pos)
        next_red_pos = lax.select(
            red_target_reached,
            self._sample_pos(next_agent_pos, state.blue_pos, sample_key),
            state.red_pos,
        )

        blue_target_positive_reward = (state.step_count // self.swap_period) % 2 == 0
        reward = lax.select(
            blue_target_positive_reward,
            lax.select(
                blue_target_reached,
                self._reward_good_target,
                lax.select(
                    red_target_reached,
                    self._reward_bad_target,
                    self._reward_no_target,
                ),
            ),
            lax.select(
                blue_target_reached,
                self._reward_bad_target,
                lax.select(
                    red_target_reached,
                    self._reward_good_target,
                    self._reward_no_target,
                ),
            ),
        )

        next_state = State(
            key=next_key,
            agent_pos=next_agent_pos,
            blue_pos=next_blue_pos,
            red_pos=next_red_pos,
            step_count=state.step_count + 1,
        )
        return next_state, reward

    def _state_to_timestep(self, state: State, reward: Array) -> TimeStep[Array]:
        """Maps an environment state to an observation. The reward is needed to output the timestep.

        Args:
            state: State of the environment after taking a step.
            reward: reward computed from updating the state.

        Returns:
            timestep containing the new observation.

        """
        next_obs = jnp.array(
            [
                nn.one_hot(
                    state.agent_pos, max(self.n_rows, self.n_cols), dtype=jnp.float32
                ),
                nn.one_hot(
                    state.blue_pos, max(self.n_rows, self.n_cols), dtype=jnp.float32
                ),
                nn.one_hot(
                    state.red_pos, max(self.n_rows, self.n_cols), dtype=jnp.float32
                ),
            ],
            dtype=jnp.float32,
        )

        timestep: TimeStep[Array] = lax.cond(
            state.step_count % (self.n_periods * 2 * self.swap_period) == 0,
            lambda _: truncation(reward=reward, observation=next_obs),
            lambda _: transition(reward=reward, observation=next_obs),
            None,
        )
        return timestep

    def _sample_pos(self, agent_pos: Array, item_pos: Array, key: PRNGKey) -> Array:
        """Internal function that samples the coordinates of a new target.

        Args:
            agent_pos: jax array (int) of shape (2,) representing the coordinates of the agent.
            item_pos: jax array (int) of shape (2,) representing the coordinates of the other
                target to avoid sampling on the same location.
            key: random key used for the sampling operation.

        Returns:
            Coordinates of the new target.

        """
        free_slots = jnp.ones((self.n_rows, self.n_cols), float)
        free_slots = free_slots.at[tuple(agent_pos)].set(0.0)
        free_slots = free_slots.at[tuple(item_pos)].set(0.0)
        pos_index = random.choice(
            key, jnp.arange(self.n_rows * self.n_cols), p=free_slots.flatten()
        )
        row = pos_index // self.n_cols
        col = pos_index % self.n_cols
        return jnp.array((row, col), int)
