from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from gym import spaces
from ray.rllib import MultiAgentEnv

EMPTY = 0
OBSTACLE = 1
SOURCE = 2
TARGET = 3
HEAD = 4

NOOP = 0
LEFT = 1
UP = 2
RIGHT = 3
DOWN = 4

TIMEOUT = 1000


@dataclass
class Agent:
    """Data Class used for tracking the agent_id, current_position
    and target_position of each agent in the environment"""

    agent_id: int
    position: Tuple[int, int]
    target: Tuple[int, int] = (-1, -1)

    @property
    def done(self) -> bool:
        return self.position == self.target


class PcbGridEnv(MultiAgentEnv):
    """PCB grid environment.
    Simplified from full version for easy testing of multi-agent strategies"""

    metadata = {
        "render.modes": ["human", "png", "rgb_array"],
        "video.frames_per_second": 2,
    }
    rows: int
    cols: int
    num_agents: int
    agents: List[Agent]
    grid: np.ndarray

    def __init__(
        self,
        rows: int,
        cols: int,
        num_agents: int,
        difficulty: str = "easy",
        reward_per_timestep: Optional[float] = None,
        reward_per_connected: float = 1.0,
        reward_per_blocked: float = -1.0,
    ):
        """A simple grid environment that represents the PCB environment.

        Args:
            rows: grid height.
            cols: grid width.
            num_agents: number of agents in the grid.
            difficulty: hard or easy layouts. If hard, there is at least 1 overlap.
            reward_per_timestep: a small negative reward provided to every agent at each timestep if
                they do not connect and are not blocked. If None, a default value of
                -1.0 / (rows + cols) is used.
            reward_per_connected: reward given if the agent connects.
            reward_per_blocked: reward given if an agent blocks itself.
        """
        self.rows = rows
        self.cols = cols
        self.num_agents = num_agents
        self.difficulty = difficulty.upper()
        self.reward_per_timestep = (
            reward_per_timestep
            if reward_per_timestep is not None
            else -1.0 / (rows + cols)
        )
        self.reward_per_connected = reward_per_connected
        self.reward_per_blocked = reward_per_blocked
        self.action_space = spaces.Discrete(5)
        self.obs_ints = 2 + 3 * num_agents
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, self.obs_ints, (rows, cols), dtype=np.int),
                "action_mask": spaces.Box(
                    0, 1, shape=(self.action_space.n,), dtype=np.int
                ),
            }
        )
        rewards = [
            self.reward_per_timestep,
            self.reward_per_connected,
            self.reward_per_blocked,
        ]
        self.reward_range = (min(rewards), max(rewards))

    def seed(self, seed: int) -> None:
        np.random.seed(seed)

    def reset(self) -> dict:
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int)
        self.agents = []
        for agent_id in range(self.num_agents):
            self._spawn_agent(agent_id)

        return {
            agent.agent_id: self._agent_observation(agent.agent_id)
            for agent in self.agents
        }

    def step(self, actions: dict) -> Tuple[dict, dict, dict, dict]:
        for agent_id, action in actions.items():
            self._step_agent(agent_id, action)

        observations = {
            agent_id: self._agent_observation(agent_id) for agent_id in actions.keys()
        }
        dones = {
            agent_id: (
                self.agents[agent_id].done
                or is_agent_blocked(observations[agent_id]["action_mask"])
            )
            for agent_id in actions.keys()
        }
        dones["__all__"] = all(dones[agent_id] for agent_id in dones.keys())

        rewards = {
            agent_id: self._agent_reward(
                agent_id, observations[agent_id]["action_mask"]
            )
            for agent_id in actions.keys()
        }

        return observations, rewards, dones, {}

    def _spawn_agent(self, agent_id: int) -> None:
        """
        Spawns an agent in a random position and gives it the ID of agent_id.

        Args:
            agent_id: ID of the agent to spawn.
        """
        agent = Agent(agent_id, self._random_empty_position())
        self.grid[agent.position] = HEAD + 3 * agent_id
        agent.target = self._random_empty_position()
        self.agents.append(agent)

        if self.difficulty == "HARD":
            for _ in range(TIMEOUT):
                if self._has_crossover(agent):
                    break
                agent.target = self._random_empty_position()

        self.grid[agent.target] = TARGET + 3 * agent_id

    def _has_crossover(self, agent: Agent) -> bool:
        """
        Check if agent path to target crosses over the path of any other agent.

        Args:
            agent: Agent whose path is to be checked.

        Returns: If the agent path crosses over that of another agent.

        """
        if len(self.agents) == 1:
            return True

        return any(
            _intersect(agent.position, agent.target, a.position, a.target)
            for a in self.agents
        )

    def _random_empty_position(self) -> Tuple[int, int]:
        """
        Generate a random empty position in the grid.

        Returns: A position in the grid.

        """
        empty = False

        while not empty:
            row, col = np.random.randint((0, 0), (self.rows, self.cols))
            pos = (row, col)
            empty = self.grid[pos] == EMPTY

        return pos

    def _agent_reward(self, agent_id: int, action_mask: np.ndarray) -> float:
        """
        Calculated the reward of the agent with ID agent_id.

        Args:
            agent_id: ID of the agent to get the rewards for.
            action_mask: the action mask of agent with ID agent_id.

        Returns: the reward for the agent with ID agent_id.
        """
        agent = self.agents[agent_id]
        if agent.done:
            return self.reward_per_connected
        elif is_agent_blocked(action_mask):
            return self.reward_per_blocked
        else:
            return self.reward_per_timestep

    def _agent_observation(self, agent_id: int) -> dict:
        """
        Get local observation for agent.

        Args:
            agent_id: ID of agent to get observation for.

        Returns: Dictionary of the observation.

        """
        obs = np.copy(self.grid)
        mask = self._get_action_mask(agent_id)
        if agent_id == 0:
            return {"image": obs, "action_mask": mask}

        d = agent_id * 3
        first_indices = np.where((obs >= 2) & (obs <= 4))
        current_indices = np.where((obs >= 2 + d) & (obs <= 4 + d))
        obs[first_indices] += d
        obs[current_indices] -= d
        return {"image": obs, "action_mask": mask}

    def _get_action_mask(self, agent_id: int) -> List[int]:
        """
        Get action mask for agent.

        Args:
            agent_id: ID of agent to get action mask for.

        Returns: List of integers representing the action mask.

        """

        def _is_free_cell(x: int, y: int, target: int) -> bool:
            is_free: bool = (
                self.grid[agent_x, y] == EMPTY or self.grid[agent_x, y] == target
            )
            return is_free

        mask = [0] * 5
        mask[0] = 1
        agent = self.agents[agent_id]
        agent_x, agent_y = agent.position
        target = self.grid[agent.target[0]][agent.target[1]]
        # left
        mask[1] = int(agent_y > 0 and _is_free_cell(agent_x, agent_y - 1, target))
        # up
        mask[2] = int(agent_x > 0 and _is_free_cell(agent_x - 1, agent_y, target))
        # right
        mask[3] = int(
            agent_y < self.cols - 1 and _is_free_cell(agent_x, agent_y + 1, target)
        )
        # down
        mask[4] = int(
            agent_x < self.rows - 1 and _is_free_cell(agent_x + 1, agent_y, target)
        )
        return mask

    def _step_agent(self, agent_id: int, action: int) -> None:
        """
        Step the agent in the environment using the given action.

        Args:
            agent_id: ID of the agent to step.
            action: Action to take.
        """
        if action == NOOP:
            return

        agent = self.agents[agent_id]
        if agent.done:
            return

        position = move(agent.position, action)
        if self._is_valid(position, agent.agent_id):
            self._move_agent(agent, position)

    def _is_valid(self, position: Tuple[int, int], agent_id: int) -> bool:
        """
        Check if it is valid for the agent with the given ID to move into the given position.

        Args:
            position: Position agent is moving into.
            agent_id: ID of moving agent.

        Returns: If the agent move is valid.

        """
        row, col = position

        return (
            0 <= row < self.rows
            and 0 <= col < self.cols
            and (
                self.grid[position] == EMPTY
                or self.grid[position] == TARGET + 3 * agent_id
            )
        )

    def _move_agent(self, agent: Agent, position: Tuple[int, int]) -> None:
        """
        Update the grid and the agent with the new position.

        Args:
            agent: Agent to update.
            position: New position.

        """
        self.grid[agent.position] = SOURCE + 3 * agent.agent_id
        self.grid[position] = HEAD + 3 * agent.agent_id
        agent.position = position

    def __repr__(self) -> str:
        return f"<PCBGridEnv(rows={self.rows}, cols={self.cols}, agents={self.num_agents})>"


def move(position: Tuple[int, int], action: int) -> Tuple[int, int]:
    row, col = position
    if action == LEFT:
        return row, col - 1
    if action == RIGHT:
        return row, col + 1
    if action == UP:
        return row - 1, col
    if action == DOWN:
        return row + 1, col
    raise ValueError(f"unsupported action '{action}'")


def _is_counter_clockwise(
    point_a: Tuple[int, int], point_b: Tuple[int, int], point_c: Tuple[int, int]
) -> bool:
    is_cc: bool = (point_c[1] - point_a[1]) * (point_b[0] - point_a[0]) > (
        point_b[1] - point_a[1]
    ) * (point_c[0] - point_a[0])
    return is_cc


def _intersect(
    point_a: Tuple[int, int],
    point_b: Tuple[int, int],
    point_c: Tuple[int, int],
    point_d: Tuple[int, int],
) -> bool:
    """
    Check if line segments AB and CD intersect.

    Args:
        point_a: first point.
        point_b: second point.
        point_c: third point.
        point_d: fourth point.

    Returns: If line segments intersect.

    """
    return _is_counter_clockwise(point_a, point_c, point_d) != _is_counter_clockwise(
        point_b, point_c, point_d
    ) and _is_counter_clockwise(point_a, point_b, point_c) != _is_counter_clockwise(
        point_a, point_b, point_d
    )


def is_agent_blocked(action_mask: np.ndarray) -> bool:
    """
    Checks if an agent is blocked given its action mask.

    Args:
        action_mask: a boolean numpy array of legal actions.

    Returns: True if any value (other than the first value) in the action_mask is True
        otherwise False.

    """
    return not np.any(action_mask[1:])
