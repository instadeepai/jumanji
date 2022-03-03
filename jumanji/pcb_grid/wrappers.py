import dm_env
import numpy as np
from dm_env import specs

from jumanji.pcb_grid import pcb_grid
from jumanji.pcb_grid.pcb_grid import PcbGridEnv


class DeepMindEnvWrapper(dm_env.Environment):
    """A wrapper that converts PcbGridEnv to dm_env.Environment."""

    _obs_agent_id = 0  # all observations from the perspective of agent 0

    def __init__(self, env: PcbGridEnv, horizon: int = None):
        """Wrapper for the PcbGridEnv in the DeepMind style. This wrapper also makes the
        environment inherently single agent by requiring that an action contains both the desired
        action and the agent which must take that action.

        Args:
            env: A PcbGrid environment to wrap to a dm_env.Environment.
            horizon: Maximal number of steps before the episode is reset. If left as `None`
             (default) horizon is automatically set as `(env.rows + env.cols) * env.num_agents`.
        """
        assert isinstance(
            env, PcbGridEnv
        ), f"Expected env of type PcbGridEnv, got {env} of type {type(env)}."

        if horizon is None:
            # rows + cols is the max distance an agent should have to travel if start and target
            # are in opposite corners of the map, and it is not blocked.
            horizon = (env.rows + env.cols) * env.num_agents

        self._env = env
        self.horizon = horizon
        self._step = 0
        assert horizon > 0, f"horizon must be positive, got {horizon}."

    def __repr__(self) -> str:
        return self._env.__repr__()

    def reset(self) -> dm_env.TimeStep:
        """Starts a new sequence and returns the first `TimeStep` of this sequence.

        Returns:
            A `TimeStep` namedtuple containing:
            step_type: A `StepType` of `FIRST`.
            reward: None
            discount: None
            observation: A 2D numpy array representing the grid, from the
                perspective of the agent with id `self._obs_agent_id`.
        """
        self._step = 0
        observation = self._env.reset()[self._obs_agent_id]["image"]
        return dm_env.restart(observation=observation)

    def step(self, agent_action: np.ndarray) -> dm_env.TimeStep:
        """Updates the environment according to the action and returns a `TimeStep`.

        Args:
            agent_action: A NumPy array, of length 2. [agent_id: int, action: int]

        Returns:
            A `TimeStep` namedtuple containing:
                step_type: A `StepType` value.
                reward: Reward for the action taken by the selected agent.
                discount: A discount, 0 if done else 1.
                observation: A 2D numpy array representing the grid, from the
                    perspective of the agent with id `self._obs_agent_id`.
        """
        self._step += 1

        # format the action
        # first element is the agent to move and second element is the action
        agent_id, action = agent_action
        actions = {agent_id: pcb_grid.NOOP for agent_id in range(self._env.num_agents)}
        actions[agent_id] = action

        observations, rewards, dones, infos = self._env.step(actions)

        # either passed time horizon or all agents blocked/finished
        done = dones["__all__"] or self._step >= self.horizon
        step_type = dm_env.StepType.MID if not done else dm_env.StepType.LAST

        observation = observations[self._obs_agent_id]["image"]
        discount = float(not done)  # 0 if done else 1
        reward = rewards[agent_id]  # only give the reward of the current agent

        return dm_env.TimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=observation,
        )

    def observation_spec(self) -> specs.Array:
        """Returns the observation spec."""
        return specs.BoundedArray(
            shape=(
                self._env.rows,
                self._env.cols,
            ),
            dtype=np.int32,
            minimum=0,
            maximum=self._env.obs_ints,
            name="observation",
        )

    def action_spec(self) -> specs.Array:
        """Returns the action spec."""
        return specs.BoundedArray(
            (2,),
            np.int32,
            minimum=(0, 0),
            maximum=(self._env.num_agents - 1, 4),
            name="action",
        )

    def unwrapped(self) -> PcbGridEnv:
        return self._env
