import os
from typing import List

import mujoco_py
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box, Dict

DEFAULT_CAMERA_CONFIG = {
    "distance": 20.0,
}

DESCRIPTORS_CLIPPING = {
    "min_x": 0.0,
    "max_x": 30.0,
    "min_y": -8.0,
    "max_y": 8.0,
}


def clip_state_descriptor(state_descriptor):
    """Clip state descriptor to restrict the descriptor space."""
    clipped_state_descriptor = np.array(
        [
            np.clip(
                state_descriptor[0],
                DESCRIPTORS_CLIPPING["min_x"],
                DESCRIPTORS_CLIPPING["max_x"],
            ),
            np.clip(
                state_descriptor[1],
                DESCRIPTORS_CLIPPING["min_y"],
                DESCRIPTORS_CLIPPING["max_y"],
            ),
        ]
    )
    return clipped_state_descriptor


class AntTrap(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    Implements the AntTrap environment where an articulated ant must run as fast as
    possible in the forward direction and avoid a trap that makes the environment
    highly deceptive.

    The state descriptors are the (x,y) position of the ant at a given timestep. Since
    the environment is not bounded by walls (unlike mazes), the descriptor space is
    clipped so that it is not infinite. State descriptors are clipped between [-8;8] on
    the y-axis (so as to let a sufficient space for the ant to bypass the trap on left
    and right) and between [0;+inf] on the x-axis
    """

    def __init__(self):
        local_path = os.path.dirname(__file__)
        xml_file = local_path + "/mujoco_assets/ant_trap.xml"
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        utils.EzPickle.__init__(self)

        self._obs_shape = self._get_obs().shape
        self.observation_space = Dict(
            {
                "observation": Box(-np.inf, np.inf, self._obs_shape),
                "state_descriptor": Box(-np.inf, np.inf, (2,)),
            }
        )

    @property
    def descriptors_min_values(self) -> List[float]:
        """Minimum values for descriptors."""
        return [DESCRIPTORS_CLIPPING["min_x"], DESCRIPTORS_CLIPPING["min_y"]]

    @property
    def descriptors_max_values(self) -> List[float]:
        """Maximum values for descriptors."""
        return [DESCRIPTORS_CLIPPING["max_x"], DESCRIPTORS_CLIPPING["max_y"]]

    @property
    def descriptors_names(self) -> List[str]:
        """Descriptors names."""
        return ["x_pos", "y_pos"]

    def reset(self):
        """Reset the environment to its initial state and returns an observation."""
        self.sim.reset()
        self.reset_model()
        xy_position_after = self.get_body_com("torso")[:2].copy()

        state_descriptor = clip_state_descriptor(xy_position_after)
        obs_dict = {
            "observation": self._get_obs(),
            "state_descriptor": state_descriptor,
        }
        return obs_dict

    def step(self, action: np.ndarray):
        """
        Make an environment step and return the observation, reward and if the step
        marks the end of an episode.
        """
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        xy_position_after = self.get_body_com("torso")[:2].copy()

        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.5 * np.square(action).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and 0.2 <= state[2] <= 1.0
        done = not notdone

        # State descriptor clipping to restrict the otherwise infinite descriptor space
        state_descriptor = clip_state_descriptor(xy_position_after)
        obs_dict = {
            "observation": self._get_obs(),
            "state_descriptor": state_descriptor,
        }

        return (
            obs_dict,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
                x_position=xy_position_after[0],
                y_position=xy_position_after[1],
            ),
        )

    def _get_obs(self):
        """
        Get an observation of the current state. Observation contains the agent position
        and velocity.

        # Note: Currently, the additional informations that are almost always zeros are
        # disabled, uncomment the last line if needed.
        """
        return np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        """
        Reset the model with some stochasticity on its initial position and velocity.
        """
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def close(self):
        """
        Close viewer.
        """
        if self.viewer is not None:
            self.viewer.finish()
            self.viewer = None

    def render(self, mode="human", width: int = 400, height: int = 400):
        """
        Gym rendering function.
        """
        if mode == "rgb_array":
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer().read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            data = data
            return data[::-1, :, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def _get_viewer(self, mode="human") -> mujoco_py.MjViewer:
        """
        Returns Mujoco viewer.
        """
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
            self.viewer_setup()
        return self.viewer

    def viewer_setup(self):
        """Setup the camera."""
        self.viewer.cam.distance = 27
        self.viewer.cam.elevation = -23.386441767692556
        self.viewer.cam.azimuth = 70
        self.viewer.cam.lookat[0] = 8.94361076
        self.viewer.cam.lookat[1] = -3.52206891
        self.viewer.cam.lookat[2] = 2.92638128
