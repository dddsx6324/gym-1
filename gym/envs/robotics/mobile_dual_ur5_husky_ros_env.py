


import os
import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding
from gym.envs.robotics import utils_ur5
from gym.envs.robotics import rotations, utils


from collections import OrderedDict

def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space



class MobileDualUR5HuskyGymEnv(robot_gym_env.RobotGymEnv):
    """Superclass for all Dual_UR5_Husky environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type, n_actions,
        use_real_robot, debug_print, use_arm, object_type,
    ):
        """Initializes a new Dual_UR5_Husky environment.
        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            n_actions : the number of actuator
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.n_actions = n_actions
        self.object_type = object_type

        self.arm_dof = 3
        self.gripper_dof = 1
        # self.n_actions = self.arm_dof + self.gripper_dof
        
        self.gripper_actual_dof = 4
        self.gripper_close = False

        self.husky_init_pos = [0,0]

        self.left_arm_joint_names = [
            'l_ur5_arm_shoulder_pan_joint',
            'l_ur5_arm_shoulder_lift_joint',
            'l_ur5_arm_elbow_joint',
            'l_ur5_arm_wrist_1_joint',
            'l_ur5_arm_wrist_2_joint',
            'l_ur5_arm_wrist_3_joint',
        ]
        self.right_arm_joint_names = [
            'r_ur5_arm_shoulder_pan_joint',
            'r_ur5_arm_shoulder_lift_joint',
            'r_ur5_arm_elbow_joint',
            'r_ur5_arm_wrist_1_joint',
            'r_ur5_arm_wrist_2_joint',
            'r_ur5_arm_wrist_3_joint',
        ]
        self.init_pos = {
            'r_ur5_arm_shoulder_pan_joint': 0.0,
            'r_ur5_arm_shoulder_lift_joint': 0.0,
            'r_ur5_arm_elbow_joint': 0.0,
            'r_ur5_arm_wrist_1_joint': 0.0,
            'r_ur5_arm_wrist_2_joint': 0.0,
            'r_ur5_arm_wrist_3_joint': 0.0,
        }

        self.debug_print = debug_print

        self._is_success = 0

        self.current_object_pos = [0,0,0]
        self.current_grip_pos = [0,0,0]

        # rospy.init_node("gym")
        self._use_real_robot = use_real_robot
        if self._use_real_robot:
            # import rospy
            from gym.envs.robotics.ros_interface import husky_ur_ros
            self.husky_ur5_robot = husky_ur_ros.HuskyUR5ROS(debug_print=debug_print, use_arm=use_arm)
            self.use_arm = use_arm

        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        self.observation_space = convert_observation_to_space(obs)

    def _set_action(self, action):

        assert action.shape == (self.n_actions,) # 6 mobile base
        action = action.copy()  # ensure that we don't change the action outside of this scope
        if self.debug_print:
            print("_set_action:", action)
        pos_ctrl, base_ctrl, gripper_ctrl = action[:3], action[3:-1], action[-1]

        pos_ctrl *= 0.05  # limit maximum change in position
        base_ctrl *= 0.5
        base_ctrl[1] = 0

        rot_ctrl = [0, 0.707, 0.707, 0] # fixed rotation of the end effector, expressed as a quaternion

        # TODO: if we cannot see the object or just part, how is that?
        if self.gripper_close:
            gripper_ctrl = -1.0
        else:
            gripper_ctrl = 1.0

        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        # action = np.concatenate([pos_ctrl, rot_ctrl, base_ctrl, gripper_ctrl])

        ee_pose = self.husky_ur5_robot.arm_get_ee_pose(self.use_arm)

        arm_action = pos_ctrl
        print("arm_action: ", arm_action)
        print("base_action: ", base_ctrl)

        # Applay action to real robot
        # self.husky_ur5_robot.arm_set_ee_pose_relative(pos_ctrl)
        self.husky_ur5_robot.arm_set_ee_pose_relative(arm_action)
        self.husky_ur5_robot.base_velocity_cmd(base_ctrl)
        # self.husky_ur5_robot.base_go_to_relative(base_ctrl)
        if self.gripper_close:
            self.husky_ur5_robot.gripper_close(self.use_arm)
        else:
            self.husky_ur5_robot.gripper_open(self.use_arm)

    def _get_obs(self):

        joint_angles = []
        joint_velocity = []
        ee_position = []
        ee_orientation = []

        if self.use_arm == 'left':
            joint_names_dict = self.husky_ur5_robot.arm_get_joint_angles(self.use_arm)
            joint_velocity_dict = self.husky_ur5_robot.arm_get_joint_velocity(self.use_arm)
            for i in self.left_arm_joint_names:
                joint_angles.append(joint_names_dict[i])
                joint_velocity.append(joint_velocity_dict[i])
            ee_pose = self.husky_ur5_robot.arm_get_ee_pose(self.use_arm)
            ee_position = [ee_pose.pose.position.x, 
                            ee_pose.pose.position.y,
                            ee_pose.pose.position.z]
            ee_orientation = [ee_pose.pose.orientation.w,                
                                ee_pose.pose.orientation.x,
                                ee_pose.pose.orientation.y,
                                ee_pose.pose.orientation.z,]

        if self.use_arm == 'right':
            joint_names_dict = self.husky_ur5_robot.arm_get_joint_angles(self.use_arm)
            joint_velocity_dict = self.husky_ur5_robot.arm_get_joint_velocity(self.use_arm)
            for i in self.right_arm_joint_names:
                joint_angles.append(joint_names_dict[i])
                joint_velocity.append(joint_velocity_dict[i])
            ee_pose = self.husky_ur5_robot.arm_get_ee_pose(self.use_arm)
            ee_position = [ee_pose.pose.position.x, 
                            ee_pose.pose.position.y,
                            ee_pose.pose.position.z]
            ee_orientation = [ee_pose.pose.orientation.w,                
                                ee_pose.pose.orientation.x,
                                ee_pose.pose.orientation.y,
                                ee_pose.pose.orientation.z,]

        grip_pos = np.array(ee_position)
        # object_pos = np.array([0.2, 0.2, 0.2])
        object_pos = np.array(self.husky_ur5_robot.get_object_position())
        self.current_grip_pos = grip_pos
        self.current_object_pos = object_pos
        
        object_rel_pos = object_pos - grip_pos
        ur5_qpos = np.array(joint_angles)
        ur5_qvel = np.array(joint_velocity)
        if self.debug_print:
            print("grip_pos: ", grip_pos)
            # print("object_pos: ", object_pos)
            print("object_rel_pos: ", object_rel_pos)
            print("ur5_qpos: ", ur5_qpos)
            print("ur5_qvel: ", ur5_qvel)
            print("is_success: ", self._is_success)
        obs = np.concatenate([
            grip_pos,
            # object_pos,
            object_rel_pos,
            ur5_qpos,
            ur5_qvel,
            [self._is_success],
        ])
        if self.debug_print:
            print("observation: ", obs)
        return obs

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.husky_ur5_robot.control_rate.sleep() # 10Hz
        obs = self._get_obs()
        info = ""

        return obs, 0, False, info

    def reset(self):
        return self._get_obs()