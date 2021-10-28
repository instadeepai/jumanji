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


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class HumanoidTrap(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    Implements the HmanoidTrap environment where an articulated humanoid must run
    as fast as possible in the forward direction and avoid a trap that makes
    the environment highly deceptive.

    The state descriptors are the (x,y) position of the ant at a given timestep. Since
    the environment is not bounded by walls (unlike mazes), the descriptor space is
    clipped so that it is not infinite. State descriptors are clipped between [-8;8] on
    the y-axis (so as to let a sufficient space for the ant to bypass the trap on left
    and right) and between [0;+inf] on the x-axis
    """

    def __init__(self):
        local_path = os.path.dirname(__file__)
        xml_file = local_path + "/mujoco_assets/humanoid_trap.xml"
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
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(action, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        # xy_position_after = np.array([self.sim.data.xipos[0], self.sim.data.xipos[1]])
        xy_position_after = self.get_body_com("torso")[:2].copy()

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
                reward_linvel=lin_vel_cost,
                reward_quadctrl=-quad_ctrl_cost,
                reward_alive=alive_bonus,
                reward_impact=-quad_impact_cost,
                x_position=xy_position_after[0],
                y_position=xy_position_after[1],
            ),
        )

    def _get_obs(self):
        """
        Get an observation of the current state. Observation contains the agent position
        and velocity.
        """
        data = self.sim.data
        return np.concatenate(
            [
                data.qpos.flat[2:],
                data.qvel.flat,
                data.cinert.flat,
                data.cvel.flat,
                data.qfrc_actuator.flat,
                data.cfrc_ext.flat,
            ]
        )

    def reset_model(self):
        """
        Reset the model with some stochasticity on its initial position and velocity.
        """
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(
                low=-c,
                high=c,
                size=self.model.nv,
            ),
        )
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
        self.viewer.cam.distance = 14
        self.viewer.cam.elevation = -23.386441767692556
        self.viewer.cam.azimuth = 70
        self.viewer.cam.lookat[0] = 2
        self.viewer.cam.lookat[1] = -3.52206891
        self.viewer.cam.lookat[2] = 2.92638128
