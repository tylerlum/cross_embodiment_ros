#!/usr/bin/env python

from pathlib import Path
import rospy
import pybullet as p
from sensor_msgs.msg import JointState
import numpy as np

NUM_ARM_JOINTS = 7
NUM_HAND_JOINTS = 16
BLUE_RGBA = [0, 0, 1, 0.5]


class FakeRobotNode:
    def __init__(self):
        # ROS setup
        rospy.init_node("fake_robot_ros_node")

        # Publisher and subscriber
        self.iiwa_pub = rospy.Publisher("/iiwa/joint_states", JointState, queue_size=10)
        self.iiwa_cmd_sub = rospy.Subscriber(
            "/iiwa/joint_cmd", JointState, self.iiwa_joint_cmd_callback
        )

        # Store the latest iiwa joint command
        self.iiwa_joint_cmd = None

        # Initialize PyBullet
        # Create a real robot (simulating real robot) and a command robot (visualizing commands)
        rospy.loginfo("~" * 80)
        rospy.loginfo("Initializing PyBullet")
        self.robot_id, self.robot_cmd_id = self.initialize_pybullet()
        rospy.loginfo("PyBullet initialized!")
        rospy.loginfo("~" * 80)

        # Set control rate to 60Hz
        self.rate = rospy.Rate(60)

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
        DEFAULT_ARM_Q = np.array(
            [
                0,
                0,
                0,
                -1.57,
                0,
                1.57,
                0,
            ]
        )
        DEFAULT_HAND_Q = np.array(
            [
                0.0,
                0.3,
                0.3,
                0.3,
                0.0,
                0.3,
                0.3,
                0.3,
                0.0,
                0.3,
                0.3,
                0.3,
                0.72383858,
                0.60147215,
                0.33795027,
                0.60845138,
            ]
        )
        assert DEFAULT_ARM_Q.shape == (
            NUM_ARM_JOINTS,
        ), f"{DEFAULT_ARM_Q.shape} != ({NUM_ARM_JOINTS},)"
        assert DEFAULT_HAND_Q.shape == (
            NUM_HAND_JOINTS,
        ), f"{DEFAULT_HAND_Q.shape} != ({NUM_HAND_JOINTS},)"
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

    def iiwa_joint_cmd_callback(self, msg):
        """Callback to update the commanded joint positions."""
        self.iiwa_joint_cmd = np.array(msg.position)

    def update_pybullet(self):
        """Update the PyBullet simulation with the commanded joint positions."""
        if self.iiwa_joint_cmd is not None:
            # Use self.iiwa_joint_cmd as the actual joint positions for the command robot
            q_cmd = np.concatenate([self.iiwa_joint_cmd, np.zeros(NUM_HAND_JOINTS)])
            self.set_robot_state(self.robot_cmd_id, q_cmd)

            # # Use self.iiwa_joint_cmd as P target for the robot
            # for joint_index in range(NUM_ARM_JOINTS):
            #     p.setJointMotorControl2(
            #         self.robot_id,
            #         joint_index,
            #         p.POSITION_CONTROL,
            #         self.iiwa_joint_cmd[joint_index],
            #         force=1,
            #     )

            # p.stepSimulation()

            # Interpolate between the current joint positions and the commanded joint positions for the real robot
            # Physics was being weird
            q_state = self.get_robot_state(self.robot_id)
            delta_q = q_cmd - q_state
            delta_q_norm = np.linalg.norm(delta_q)
            max_delta_q_norm = 0.1
            # from live_plotter import FastLivePlotter
            from live_plotter import LivePlotter

            if not hasattr(self, "plotter"):
                # self.plotter = FastLivePlotter(titles=["Delta Q Norm"], xlabels=["Time"], ylabels=["Norm"])
                self.plotter = LivePlotter(
                    default_titles=["Delta Q Norm"],
                    default_xlabels=["Time"],
                    default_ylabels=["Norm"],
                )
                self.delta_q_norms = []
            self.delta_q_norms.append(delta_q_norm)
            self.plotter.plot(y_data_list=[np.array(self.delta_q_norms)])
            if delta_q_norm > max_delta_q_norm:
                delta_q = max_delta_q_norm * delta_q / delta_q_norm
            self.set_robot_state(self.robot_id, q_state + delta_q)
        else:
            rospy.loginfo("Still waiting for joint command...")

    def publish_joint_states(self):
        """Publish the current joint states from PyBullet."""
        q = self.get_robot_state(self.robot_id)

        arm_joint_states = JointState()
        arm_joint_states.header.stamp = rospy.Time.now()
        arm_joint_states.name = ["joint_" + str(i) for i in range(NUM_ARM_JOINTS)]
        arm_joint_states.position = [q[i] for i in range(NUM_ARM_JOINTS)]

        self.iiwa_pub.publish(arm_joint_states)

    def run(self):
        """Main loop to run the node, update simulation, and publish joint states."""
        while not rospy.is_shutdown():
            # Update the PyBullet simulation with the current joint commands
            self.update_pybullet()

            # Publish the current joint states to ROS
            self.publish_joint_states()

            # Sleep to maintain the loop rate
            self.rate.sleep()

        # Disconnect from PyBullet when shutting down
        p.disconnect()


if __name__ == "__main__":
    try:
        # Create and run the FakeRobotNode
        node = FakeRobotNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
