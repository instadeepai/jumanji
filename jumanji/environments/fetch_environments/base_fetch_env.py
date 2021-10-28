import copy
import os
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gym
import mujoco_py
import numpy as np
from gym import spaces
from gym.envs.robotics import rotations, utils

# Custom types
Observation = Dict[str, np.array]
Action = np.array
Reward = float
Done = bool
Info = Dict
EnvironmentState = Tuple
Color = str
# Color = Literal['blue', 'orange']  # todo: needs python 3.8 to use Literal

# Simulation variables
BLOCK_RADIUS = 0.025
ORANGE_ZONE = {
    "x_min": 1.32,
    "x_max": 1.52,
    "y_min": 0.8,
    "y_max": 1.1,
    "z": 0.43,
}
BLUE_ZONE = {
    "x_min": 1.32,
    "x_max": 1.52,
    "y_min": 0.4,
    "y_max": 0.7,
    "z": 0.43,
}


@dataclass
class EntityState:
    """
    State of an entity (robotic arm or block).
    """

    linear_position: np.array
    linear_velocity: np.array
    angular_position: Optional[np.array]
    angular_velocity: Optional[np.array]

    @property
    def features_vector(self) -> np.array:
        """Return vector of features to represent entity."""
        features = np.concatenate(
            [value for value in self.__dict__.values() if value is not None], axis=0
        )
        return features


def add_blocks_in_xml(base_xml_path: str, out_xml_path: str, num_blocks: int):
    """
    Takes a path to a base XML file that describes the table and the robotic arm and
    adds blocks inside. The resulting xml file is saved at specified path.
    """
    tree = ET.parse(base_xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    assert num_blocks <= 10, "cant specify more than 10 blocks"

    for i in range(num_blocks):

        color = "orange" if i % 2 == 0 else "blue"

        block = ET.SubElement(
            worldbody,
            "body",
            attrib={
                "name": f"object{i}",
                "pos": f"{BLOCK_RADIUS} {BLOCK_RADIUS} {BLOCK_RADIUS}",
            },
        )

        ET.SubElement(
            block,
            "joint",
            attrib={"name": f"object{i}:joint", "type": "free", "damping": "0.01"},
        )

        ET.SubElement(
            block,
            "geom",
            attrib={
                "size": f"{BLOCK_RADIUS} {BLOCK_RADIUS} {BLOCK_RADIUS}",
                "type": "box",
                "condim": "3",
                "name": f"object{i}",
                "material": f"cube_mat_{i}",
                "class": f"object:{color}",
                "mass": "2",
            },
        )

        ET.SubElement(
            block,
            "site",
            attrib={
                "name": f"object{i}",
                "pos": "0 0 0",
                "size": "0.02 0.02 0.02",
                "type": "sphere",
            },
        )

    tree.write(out_xml_path)


def compute_position_distance(pos_a: np.array, pos_b: np.array):
    """
    Compute euclidean distance between two positions.
    """
    assert pos_a.shape == pos_b.shape
    return np.linalg.norm(pos_a - pos_b, axis=-1)


def check_stacking(
    object1_state: EntityState,
    object2_state: EntityState,
    tolerance=0.05,  # todo: remove hardcorded value
) -> bool:
    """
    Check if object1 is stacked on object2.
    """
    object1_pos = object1_state.linear_position
    object2_pos = object2_state.linear_position
    stacking_pos = np.copy(object2_pos) + np.array([0.0, 0.0, 2 * BLOCK_RADIUS])
    distance = compute_position_distance(object1_pos, stacking_pos)
    return distance < tolerance


def check_object_in_zone(object_state: EntityState, color: Color) -> bool:
    """
    Check if an object is in a color zone.
    """
    # check if object is in zone of color
    position = object_state.linear_position
    x, y = position[0], position[1]
    bounds = ORANGE_ZONE if color == "orange" else BLUE_ZONE
    cond = bounds["x_min"] <= x <= bounds["x_max"]
    cond &= bounds["y_min"] <= y <= bounds["y_max"]
    # check object is on the table
    # (otherwise being in the air above the zone works also)
    cond &= check_object_on_table(object_state)
    return cond


def check_object_on_table(object_state: EntityState, tolerance: float = 0.01) -> bool:
    """
    Check if an object lies on the table.
    """
    # check if the object is at table level (z ~ 0.43)
    position = object_state.linear_position
    delta_z = np.linalg.norm(position[2] - BLUE_ZONE["z"])
    cond = delta_z < tolerance
    return cond


def check_object_static(
    object_state: EntityState, objects_states: List[EntityState]
) -> bool:
    """
    Check if a block is in equilibrium, i.e. if it either lies on the table
    or it is stacked on other block.
    """
    # check if the object is static = at table level (z ~ 0.43)
    # or stacked onto an other object
    cond = check_object_on_table(object_state)
    for other_object_state in objects_states:
        cond |= check_stacking(object_state, other_object_state)
    return cond


def check_collisions_for_pos(
    pos: np.array,
    objects_states: List[EntityState],
    tolerance=0.07,  # todo: remove hardcorded value
) -> bool:
    """
    Check if an object at pos would collide with others in 3D. Needs to provide
    to the function the list of all blocks states.
    """
    # check collision between an object that would at pos and the other objects
    # return true if there is no collision, false otherwise
    bool = True
    for object_state in objects_states:
        object_pos = object_state.linear_position
        bool &= np.linalg.norm(object_pos - pos) > tolerance
    return bool


def check_pos_is_movable(
    pos: np.array,
    objects_states: List[EntityState],
    tolerance=0.03,  # todo: remove hardcorded value
) -> bool:
    """
    Check that there is not other object too close so that the object can
    be moved without moving others next to it or on top,
    does not look at objects that could be under
    return True if object is movable
    """
    dx, dy, dz = 2 * BLOCK_RADIUS, 2 * BLOCK_RADIUS, 2 * BLOCK_RADIUS
    increments = [
        dx * np.array([1.0, 0.0, 0.0]),
        dx * np.array([-1.0, 0.0, 0.0]),
        dy * np.array([0.0, 1.0, 0.0]),
        dy * np.array([0.0, -1.0, 0.0]),
        dz * np.array([0.0, 0.0, 1.0]),
    ]
    can_be_moved = True
    for increment in increments:
        incr_pos = pos + increment
        can_be_moved &= check_collisions_for_pos(
            pos=incr_pos, objects_states=objects_states, tolerance=tolerance
        )
    return can_be_moved


class BaseMultiTaskFetchEnv(gym.Env, gym.utils.EzPickle):
    """
    Base Multi Task Fetch environment. In this environment, a robotic arm is facing
    a table with two color zones (orange and blue). There are also blocks ( from 1
    to 10) on the table. Blocks have a number between 0 and 9 as well as a color
    (orange or blue). Tasks of interest consist in sorting blocks, stacking blocks,
    moving blocks to given zone in z given order, etc ...
    This class only implements the dynamic and function to asses relationships
    between entities (for instance is block 2 stacked on block3, is block4 on zone
    orange, etc ...). Rewards always equal 0 and observations do not contain goals.
    For non zero rewards and goals use environments that wrap this one.
    """

    def __init__(
        self,
        num_blocks: int = 4,
        base_xml_path: str = "assets/fetch/multi_task_fetch_arm_base.xml",
    ):

        self._num_blocks = num_blocks

        self._gripper_extra_height = 0.2
        self._distance_threshold = 0.015

        self._init_object_in_gripper = False
        self._init_object_under_gripper = False
        self._init_stack_probability = 0.0
        self._object_idx = None

        # create model with blocks inside
        base_xml_path = str(Path(__file__).parent / base_xml_path)
        model_path = f"assets/fetch/model_with_{num_blocks}_blocks.xml"
        model_path = str(Path(__file__).parent / model_path)
        add_blocks_in_xml(
            base_xml_path=base_xml_path, out_xml_path=model_path, num_blocks=num_blocks
        )

        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)

        if not os.path.exists(fullpath):
            raise IOError("File {} does not exist".format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        n_substeps = 20
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None

        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        # initial qpos setup
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        num_objects = self._num_blocks
        for i in range(num_objects):
            initial_qpos["object" + str(i) + ":joint"] = [
                1.25,
                0.53,
                0.45,
                1.0,
                0.0,
                0.0,
                0.0,
            ]

        self._env_setup(initial_qpos=initial_qpos)
        self._initial_state = copy.deepcopy(self.sim.get_state())

        obs = self.get_obs()
        self._observation_space = spaces.Dict(
            dict(
                [
                    (
                        key,
                        spaces.Box(-np.inf, np.inf, shape=value.shape, dtype="float32"),
                    )
                    for key, value in obs.items()
                ]
            )
        )

        gym.utils.EzPickle.__init__(self)

    @property
    def dt(self) -> float:
        """
        Simulation time step.
        """
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    @property
    def observation_space(self) -> spaces.Dict:
        """
        Environment observation space.
        """
        return self._observation_space

    @property
    def action_space(self) -> spaces.Box:
        """
        Environment action space.
        """
        return spaces.Box(-1.0, 1.0, shape=(4,), dtype="float32")

    def step(self, action: Action) -> (Observation, Reward, Done, Info):
        """
        Perform one step in the environment. This environment only implements dynamics.
        It always returns rewards that equal 0 and empty into dictionaries. Non Zero
        rewards should be implemented in environments that wrap this one.
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        obs = self.get_obs()

        done = False
        info = {}
        reward = 0.0
        return obs, reward, done, info

    def _set_init_attributes(
        self,
        init_object_in_gripper_prob: float,
        init_object_under_gripper_prob: float,
        init_stack_probability: float,
    ):
        """
        Initialize environment variables.
        """
        self._init_stack_probability = init_stack_probability
        if np.random.rand() < init_object_in_gripper_prob:
            self._init_object_in_gripper = True
            self._init_object_under_gripper = True
        elif (
            np.random.rand()
            < init_object_in_gripper_prob + init_object_under_gripper_prob
        ):
            self._init_object_in_gripper = False
            self._init_object_under_gripper = True
        else:
            self._init_object_in_gripper = False
            self._init_object_under_gripper = False

    def reset(
        self,
        object_idx: Optional[int] = None,
        init_object_in_gripper_prob: float = 0.2,
        init_object_under_gripper_prob: float = 0.4,
        init_stack_probability: float = 0.4,
    ) -> Observation:
        """
        Reset env simulation. Reset objects and gripper position.
        """
        if object_idx:
            self._object_idx = object_idx
        else:
            self._object_idx = int(np.random.randint(low=0, high=self._num_blocks))

        # decide where to initialize object_idx
        self._set_init_attributes(
            init_object_in_gripper_prob,
            init_object_under_gripper_prob,
            init_stack_probability,
        )
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        obs = self.get_obs()
        return obs

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
            self._viewer_setup()
        return self.viewer

    def get_state(self) -> EnvironmentState:
        """
        Returns current simulation state.
        """
        sim_state = self.sim.get_state()
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        state = (
            sim_state.time,
            position,
            velocity,
            sim_state.act,
            sim_state.udd_state,
            self.goal.copy(),
            self.goal_to_render.copy(),
        )
        return state

    def reset_to_state(self, state: EnvironmentState):
        """
        Set simulation state.
        """
        time, qpos, qvel, act, udd_state, goal, goal_to_render = deepcopy(state)
        new_state = mujoco_py.MjSimState(time, qpos, qvel, act, udd_state)
        self.sim.set_state(new_state)
        self.goal = goal
        self.goal_to_render = goal_to_render
        self.sim.forward()

    def _set_action(self, action: Action):
        """
        Apply action in simulation.
        """
        assert action.shape == (4,)
        action = (
            action.copy()
        )  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [
            1.0,
            0.0,
            1.0,
            0.0,
        ]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def get_objects_states(self) -> Dict[str, EntityState]:
        """
        Compute all entities (gripper and blocks) positions, angular positions,
        velocities  and angular velocities and return corresponding dict.
        """

        objects_states = {}
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep

        # compute blocks states
        for i in range(self._num_blocks):
            object_pos = self.sim.data.get_site_xpos("object{}".format(i))
            object_rot = rotations.mat2euler(
                self.sim.data.get_site_xmat("object{}".format(i))
            )
            object_velp = self.sim.data.get_site_xvelp("object{}".format(i)) * dt
            object_velr = self.sim.data.get_site_xvelr("object{}".format(i)) * dt

            object_state = EntityState(
                linear_position=object_pos,
                linear_velocity=object_velp,
                angular_position=object_rot,
                angular_velocity=object_velr,
            )

            objects_states[f"block_{i}"] = object_state

        # compute robotic arm state
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        gripper_pos = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt
        arm_pos = np.concatenate([grip_pos, gripper_pos], axis=0)
        arm_vel = np.concatenate([grip_velp, gripper_vel], axis=0)

        # no rotations allowed for robotic arm
        arm_state = EntityState(
            linear_position=arm_pos,
            linear_velocity=arm_vel,
            angular_position=None,
            angular_velocity=None,
        )
        objects_states["arm"] = arm_state

        return objects_states

    def get_obs(self) -> Observation:
        """
        Compute env observation. It contains all objects positions, velocity and
        angular velocities as well as their relative pos to gripper. It also contains
        the gripper position, velocity and angular velocity.
        Compute also the achieved goal.
        """

        objects_states = self.get_objects_states()
        observation = dict(
            [(key, value.features_vector) for key, value in objects_states.items()]
        )
        return observation

    def _viewer_setup(self):
        """
        Setup Mujoco viewer.
        """
        body_id = self.sim.model.body_name2id("robot0:gripper_link")
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.0
        self.viewer.cam.elevation = -14.0

    def _init_objects_2d_pos(
        self,
        objects_2d_pos: Dict[int, Optional[np.array]],
        radius: float = 0.1,  # todo: remove hardcorded value
    ) -> Dict[int, np.array]:
        """
        Takes a dictionary that maps blocks id toward optional positions and place
        blocks for which the position equals None on the table so as to avoid
        collisions (measured with a radius). Returns a dict that contains
        randomized positions for all blocks.
        """
        # safety margin
        margin = 0.15
        # init_zone
        x_min, x_max = 0.99 + margin, 1.57 - margin
        y_min, y_max = 0.37 + margin, 1.13 - margin
        # initialize positions
        objects_to_place_idx = []
        init_positions = []
        for key, value in objects_2d_pos.items():
            if value is None:
                # object is to place
                objects_to_place_idx.append(key)
            else:
                # object already placed
                init_positions.append(value)

        # place objects not already placed
        for idx in objects_to_place_idx:
            while True:
                # sample position
                pos = np.random.uniform(
                    low=np.array([x_min, y_min]), high=np.array([x_max, y_max])
                )
                no_collision = True
                # check collision with previously placed objects
                if len(init_positions) >= 1:
                    for other_pos in init_positions:
                        no_collision &= np.linalg.norm(pos - other_pos) > radius
                # if there is no collision, save position and place next object
                if no_collision:
                    init_positions.append(pos)
                    objects_2d_pos[idx] = pos
                    break

        # return all objects positions
        return objects_2d_pos

    def _reset_sim(self):
        """
        Reset the simulations. Notably, randomize objects positions and
        gripper position.
        """
        self.sim.set_state(self._initial_state)
        objects_2d_pos = dict((idx, None) for idx in range(self._num_blocks))
        # Randomize initial gripper position
        self._reset_gripper_pos()

        # Place object id x,y and gripper
        if self._init_object_under_gripper:
            pos = self.sim.data.get_site_xpos("robot0:grip")[:2]
            objects_2d_pos[self._object_idx] = pos

        # Place objects in 2d
        objects_2d_pos = self._init_objects_2d_pos(objects_2d_pos)

        # Open gripper
        if self._init_object_in_gripper:
            self.step(action=np.array([0.0, 0.0, 0.0, 1.0]))

        # Compute final object 3d pos
        objects_qpos = dict((idx, None) for idx in range(self._num_blocks))
        for idx in range(self._num_blocks):
            object_qpos = self.sim.data.get_joint_qpos("object{}:joint".format(idx))
            object_qpos[:2] = objects_2d_pos[idx]
            object_qpos[-3:] = 0
            if idx != self._object_idx:
                object_qpos[2] = self.table_height
            else:
                if self._init_object_in_gripper:
                    object_qpos[2] = (
                        self.sim.data.get_site_xpos("robot0:grip")[2] + 0.03
                    )
                else:
                    object_qpos[2] = self.table_height
            objects_qpos[idx] = object_qpos

        # Determine if objects are stacked or not
        if self._init_object_in_gripper:
            # When object_idx starts in gripper, we dont try to stack it
            ids = np.array(
                [idx for idx in range(self._num_blocks) if idx != self._object_idx]
            )
        else:
            ids = np.array([idx for idx in range(self._num_blocks)])
        np.random.shuffle(ids)
        # check there are at least 2 objects to stack
        if len(ids) >= 2:
            for idx, idx_to_stack in zip(ids[:-1], ids[1:]):
                # randomly decide to stack on previous object or not
                if np.random.rand() < self._init_stack_probability:
                    height = objects_qpos[idx][2] + 2 * BLOCK_RADIUS
                    objects_qpos[idx_to_stack][2] = height
                    objects_qpos[idx_to_stack][:2] = objects_qpos[idx][:2]

        # Set objects positions
        for idx in range(self._num_blocks):
            self.sim.data.set_joint_qpos(
                "object{}:joint".format(idx), objects_qpos[idx]
            )

        # Close gripper
        if self._init_object_in_gripper:
            self.step(action=np.array([0.0, 0.0, 0.0, -1.0]))

        self.sim.forward()
        return True

    def _env_setup(self, initial_qpos: Dict[str, float]):
        """
        Util function to setup the simulation.
        """
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        height = -0.431 + self._gripper_extra_height
        self.central_pos = np.array([-0.580, 0.005, height])
        self.central_pos += self.sim.data.get_site_xpos("robot0:grip")
        self.central_gripper = self.sim.data.get_site_xpos("robot0:grip").copy()

        # Set height of table
        self.table_height = 0.42599082

    def _reset_gripper_pos(self):
        """
        Reset the gripper position in the simulation.
        """
        gripper_offset = np.random.uniform(
            np.array([-0.700, -0.3, -0.431]), np.array([-0.250, 0.3, -0.231])
        )
        gripper_offset += np.array([0.0, 0.0, self._gripper_extra_height])
        gripper_target = gripper_offset + self.central_gripper
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
        for _ in range(10):
            self.sim.step()
