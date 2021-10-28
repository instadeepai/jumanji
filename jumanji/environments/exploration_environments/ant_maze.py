import logging
import math
import os
from typing import List, Optional, Tuple

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box, Dict
from mujoco_py import MjViewer
from numpy.typing import ArrayLike

DESCRIPTORS_BOUNDS = {
    "min_x": -30.0,
    "max_x": 40.0,
    "min_y": -30.0,
    "max_y": 40.0,
}


class AntMaze(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    Implements the AntMaze environment where an articulated ant must find the exit of
    the maze. The reward is computed as minus the distance between the ant center
    of gravity and the maze exit that makes the environment which makes the environment
    deceptive. The state descriptors are the (x,y) position of the ant at a
    given timestep.
    """

    def __init__(self) -> None:
        self._maze_exit = np.array([35, -25])
        self._logger = logging.getLogger(f"{__name__}.AntMazeEnvironment")
        self._best_performance = -math.inf

        local_path = os.path.dirname(__file__)
        xml_file = local_path + "/mujoco_assets/ant_maze.xml"
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        utils.EzPickle.__init__(self)

        self._obs_shape = self._get_obs().shape
        self.observation_space = Dict(
            {
                "observation": Box(-np.inf, np.inf, self._obs_shape),
                "state_descriptor": Box(-np.inf, np.inf, (2,)),
            }
        )
        self.viewer: Optional[MjViewer] = None

    @property
    def descriptors_min_values(self) -> List[float]:
        """Minimum values for descriptors."""
        return [DESCRIPTORS_BOUNDS["min_x"], DESCRIPTORS_BOUNDS["min_y"]]

    @property
    def descriptors_max_values(self) -> List[float]:
        """Maximum values for descriptors."""
        return [DESCRIPTORS_BOUNDS["max_x"], DESCRIPTORS_BOUNDS["max_y"]]

    @property
    def descriptors_names(self) -> List[str]:
        """Descriptors names."""
        return ["x_pos", "y_pos"]

    def reset(self) -> dict:
        """Reset the environment to its initial state and returns an observation."""
        self.sim.reset()
        self.reset_model()
        xy_position = self.data.qpos[:2]

        obs_dict = {
            "observation": self._get_obs(),
            "state_descriptor": xy_position,
        }

        return obs_dict

    def step(self, action: ArrayLike) -> Tuple:
        self.do_simulation(action, self.frame_skip)
        distance_to_goal = np.sqrt(
            np.sum(np.square(self.data.qpos[:2] - self._maze_exit))
        )
        reward = -0.05 * distance_to_goal
        done = False
        xy_position = self.data.qpos[:2]

        self._best_performance = max(self._best_performance, -distance_to_goal)

        obs_dict = {
            "observation": self._get_obs(),
            "state_descriptor": xy_position,
        }

        return (
            obs_dict,
            reward,
            done,
            dict(
                x_position=xy_position[0],
                y_position=xy_position[1],
            ),
        )

    def _get_obs(self) -> ArrayLike:
        qpos = self.data.qpos.flatten()
        qpos[:2] = (qpos[:2] - 5) / 70
        return np.concatenate(
            [
                qpos,
                self.data.qvel.flat,
            ]
        )

    def reset_model(self) -> ArrayLike:
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self) -> None:
        if self.viewer is None:
            return

        self.viewer.cam.distance = self.model.stat.extent * 0.8
        self.viewer.cam.elevation = -45
        self.viewer.cam.lookat[0] = 4.2
        self.viewer.cam.lookat[1] = 0
