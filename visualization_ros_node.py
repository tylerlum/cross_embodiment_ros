#!/usr/bin/env python

from pathlib import Path
from typing import Literal

import numpy as np
import pybullet as p
import rospy
from sensor_msgs.msg import JointState

NUM_ARM_JOINTS = 7
NUM_HAND_JOINTS = 16
BLUE_RGBA = [0, 0, 1, 0.5]


class VisualizationNode:
    def __init__(self):
        # ROS setup
        rospy.init_node("visualization_ros_node")

        # ROS msgs
        self.iiwa_joint_cmd = None
        self.allegro_joint_cmd = None
        self.iiwa_joint_state = None
        self.allegro_joint_state = None

        # Publisher and subscriber
        self.iiwa_sub = rospy.Subscriber(
            "/iiwa/joint_states", JointState, self.iiwa_joint_state_callback
        )
        self.allegro_sub = rospy.Subscriber(
            "/allegro/joint_states", JointState, self.allegro_joint_state_callback
        )
        self.iiwa_cmd_sub = rospy.Subscriber(
            "/iiwa/joint_cmd", JointState, self.iiwa_joint_cmd_callback
        )
        self.allegro_cmd_sub = rospy.Subscriber(
            "/allegro/joint_cmd", JointState, self.allegro_joint_cmd_callback
        )

        # Initialize PyBullet
        # Create a real robot (simulating real robot) and a command robot (visualizing commands)
        rospy.loginfo("~" * 80)
        rospy.loginfo("Initializing PyBullet")
        self.robot_id, self.robot_cmd_id = self.initialize_pybullet()
        rospy.loginfo("PyBullet initialized!")
        rospy.loginfo("~" * 80)

        # Set control rate to 60Hz
        self.rate_hz = 60
        self.rate = rospy.Rate(self.rate_hz)

    def initialize_pybullet(self):
        """Initialize PyBullet, set up camera, and load the robot URDF."""
        p.connect(p.GUI)

        # Load robot URDF with a fixed base
        urdf_path = Path(
            "/juno/u/tylerlum/github_repos/bidexhands_isaacgymenvs/assets/urdf/kuka_allegro_description/kuka_allegro.urdf"
        )
        assert urdf_path.exists(), f"URDF file not found: {urdf_path}"
        robot_id = p.loadURDF(str(urdf_path), useFixedBase=True)
        robot_cmd_id = p.loadURDF(str(urdf_path), useFixedBase=True)

        # Make the robot blue
        # Change the color of each link (including the base)
        num_joints = p.getNumJoints(robot_id)
        for link_index in range(-1, num_joints):  # -1 is for the base
            p.changeVisualShape(robot_cmd_id, link_index, rgbaColor=BLUE_RGBA)

        # Set the robot to a default pose
        DEFAULT_ARM_Q = np.zeros(NUM_ARM_JOINTS)
        DEFAULT_HAND_Q = np.zeros(NUM_HAND_JOINTS)
        assert DEFAULT_ARM_Q.shape == (NUM_ARM_JOINTS,)
        assert DEFAULT_HAND_Q.shape == (NUM_HAND_JOINTS,)
        DEFAULT_Q = np.concatenate([DEFAULT_ARM_Q, DEFAULT_HAND_Q])
        self.set_robot_state(robot_id, DEFAULT_Q)
        self.set_robot_state(robot_cmd_id, DEFAULT_Q)

        # Set the camera parameters
        self.set_pybullet_camera()

        # Set gravity for simulation
        p.setGravity(0, 0, -9.81)
        return robot_id, robot_cmd_id

    def set_pybullet_camera(
        self,
        cameraDistance=2,
        cameraYaw=90,
        cameraPitch=-15,
        cameraTargetPosition=[0, 0, 0],
    ):
        """Configure the PyBullet camera view."""
        p.resetDebugVisualizerCamera(
            cameraDistance=cameraDistance,
            cameraYaw=cameraYaw,
            cameraPitch=cameraPitch,
            cameraTargetPosition=cameraTargetPosition,
        )

    def set_robot_state(self, robot, q: np.ndarray) -> None:
        assert q.shape == (23,)

        num_total_joints = p.getNumJoints(robot)
        assert num_total_joints == 27, f"num_total_joints: {num_total_joints}"
        actuatable_joint_idxs = [
            i
            for i in range(num_total_joints)
            if p.getJointInfo(robot, i)[2] != p.JOINT_FIXED
        ]
        num_actuatable_joints = len(actuatable_joint_idxs)
        assert (
            num_actuatable_joints == 23
        ), f"num_actuatable_joints: {num_actuatable_joints}"

        for i, joint_idx in enumerate(actuatable_joint_idxs):
            p.resetJointState(robot, joint_idx, q[i])

    def get_robot_state(self, robot) -> np.ndarray:
        num_total_joints = p.getNumJoints(robot)
        assert num_total_joints == 27, f"num_total_joints: {num_total_joints}"
        actuatable_joint_idxs = [
            i
            for i in range(num_total_joints)
            if p.getJointInfo(robot, i)[2] != p.JOINT_FIXED
        ]
        num_actuatable_joints = len(actuatable_joint_idxs)
        assert (
            num_actuatable_joints == 23
        ), f"num_actuatable_joints: {num_actuatable_joints}"

        q = np.zeros(num_actuatable_joints)
        for i, joint_idx in enumerate(actuatable_joint_idxs):
            q[i] = p.getJointState(robot, joint_idx)[0]  # Joint position
        return q

    def iiwa_joint_cmd_callback(self, msg: JointState):
        """Callback to update the commanded joint positions."""
        self.iiwa_joint_cmd = np.array(msg.position)

    def allegro_joint_cmd_callback(self, msg: JointState):
        """Callback to update the commanded joint positions."""
        self.allegro_joint_cmd = np.array(msg.position)

    def iiwa_joint_state_callback(self, msg: JointState):
        """Callback to update the current joint positions."""
        self.iiwa_joint_state = np.array(msg.position)

    def allegro_joint_state_callback(self, msg: JointState):
        """Callback to update the current joint positions."""
        self.allegro_joint_state = np.array(msg.position)

    def update_pybullet(self):
        """Update the PyBullet simulation with the commanded joint positions."""
        if self.iiwa_joint_cmd is None:
            rospy.logwarn("iiwa_joint_cmd is None")
            iiwa_joint_cmd = np.zeros(NUM_ARM_JOINTS)
        else:
            iiwa_joint_cmd = self.iiwa_joint_cmd

        if self.allegro_joint_cmd is None:
            rospy.logwarn("allegro_joint_cmd is None")
            allegro_joint_cmd = np.zeros(NUM_HAND_JOINTS)
        else:
            allegro_joint_cmd = self.allegro_joint_cmd

        if self.iiwa_joint_state is None:
            rospy.logwarn("iiwa_joint_state is None")
            iiwa_joint_state = np.zeros(NUM_ARM_JOINTS)
        else:
            iiwa_joint_state = self.iiwa_joint_state

        if self.allegro_joint_state is None:
            rospy.logwarn("allegro_joint_state is None")
            allegro_joint_state = np.zeros(NUM_HAND_JOINTS)
        else:
            allegro_joint_state = self.allegro_joint_state

        # Command Robot: Set the commanded joint positions
        q_cmd = np.concatenate([iiwa_joint_cmd, allegro_joint_cmd])
        q_state = np.concatenate([iiwa_joint_state, allegro_joint_state])
        self.set_robot_state(self.robot_cmd_id, q_cmd)
        self.set_robot_state(self.robot_id, q_state)

    def run(self):
        """Main loop to run the node, update simulation, and publish joint states."""
        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            # Update the PyBullet simulation with the current joint commands
            self.update_pybullet()

            # Sleep to maintain the loop rate
            before_sleep_time = rospy.Time.now()
            self.rate.sleep()
            after_sleep_time = rospy.Time.now()
            rospy.loginfo(
                f"Max rate: {1 / (before_sleep_time - start_time).to_sec()} Hz ({(before_sleep_time - start_time).to_sec() * 1000}ms), Actual rate: {1 / (after_sleep_time - start_time).to_sec()} Hz"
            )

        # Disconnect from PyBullet when shutting down
        p.disconnect()


if __name__ == "__main__":
    try:
        # Create and run the FakeRobotNode
        node = VisualizationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
