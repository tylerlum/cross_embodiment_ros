#!/usr/bin/env python

from pathlib import Path
from typing import Optional

import numpy as np
import pybullet as p
import rospy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from fabric_world import world_dict_robot_frame

NUM_ARM_JOINTS = 7
NUM_HAND_JOINTS = 16
BLUE_TRANSLUCENT_RGBA = [0, 0, 1, 0.5]
RED_TRANSLUCENT_RGBA = [1, 0, 0, 0.2]

BLUE_RGB = [0, 0, 1]
RED_RGB = [1, 0, 0]
GREEN_RGB = [0, 1, 0]
YELLOW_RGB = [1, 1, 0]
CYAN_RGB = [0, 1, 1]
MAGENTA_RGB = [1, 0, 1]
WHITE_RGB = [1, 1, 1]
BLACK_RGB = [0, 0, 0]

BLUE_RGBA = [*BLUE_RGB, 1]
RED_RGBA = [*RED_RGB, 1]
GREEN_RGBA = [*GREEN_RGB, 1]
YELLOW_RGBA = [*YELLOW_RGB, 1]
CYAN_RGBA = [*CYAN_RGB, 1]
MAGENTA_RGBA = [*MAGENTA_RGB, 1]
WHITE_RGBA = [*WHITE_RGB, 1]
BLACK_RGBA = [*BLACK_RGB, 1]


def add_cuboid(halfExtents, position, orientation, rgbaColor=RED_TRANSLUCENT_RGBA):
    # Create a visual shape for the cuboid
    visualShapeId = p.createVisualShape(
        shapeType=p.GEOM_BOX, halfExtents=halfExtents, rgbaColor=rgbaColor
    )  # Red color

    # Create a collision shape for the cuboid
    collisionShapeId = p.createCollisionShape(
        shapeType=p.GEOM_BOX, halfExtents=halfExtents
    )

    # Create the cuboid as a rigid body
    cuboidId = p.createMultiBody(
        baseMass=1,  # Mass of the cuboid
        baseCollisionShapeIndex=collisionShapeId,
        baseVisualShapeIndex=visualShapeId,
        basePosition=position,
        baseOrientation=orientation,
    )
    return cuboidId


def create_transform(
    pos: np.ndarray,
    rot: np.ndarray,
) -> np.ndarray:
    assert pos.shape == (3,)
    assert rot.shape == (3, 3)
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def add_line(start, end, rgbColor=WHITE_RGB, lineWidth=3):
    return p.addUserDebugLine(start, end, lineColorRGB=rgbColor, lineWidth=lineWidth)


def move_line(lineId, start, end, rgbColor=WHITE_RGB, lineWidth=3):
    p.addUserDebugLine(
        start,
        end,
        replaceItemUniqueId=lineId,
        lineColorRGB=rgbColor,
        lineWidth=lineWidth,
    )


def visualize_transform(
    xyz: np.ndarray,
    rotation_matrix: np.ndarray,
    length: float = 0.2,
    lines: Optional[list] = None,
) -> list:
    T = create_transform(pos=xyz, rot=rotation_matrix)
    assert T.shape == (4, 4), T.shape

    origin = np.array([0, 0, 0])
    x_pos = np.array([length, 0, 0])
    y_pos = np.array([0, length, 0])
    z_pos = np.array([0, 0, length])

    tranformed_origin = T[:3, :3] @ origin + T[:3, 3]
    tranformed_x_pos = T[:3, :3] @ x_pos + T[:3, 3]
    tranformed_y_pos = T[:3, :3] @ y_pos + T[:3, 3]
    tranformed_z_pos = T[:3, :3] @ z_pos + T[:3, 3]

    if lines is None:
        lines = []

        lines.append(add_line(tranformed_origin, tranformed_x_pos, rgbColor=RED_RGB))
        lines.append(add_line(tranformed_origin, tranformed_y_pos, rgbColor=GREEN_RGB))
        lines.append(add_line(tranformed_origin, tranformed_z_pos, rgbColor=BLUE_RGB))
        return lines
    else:
        move_line(
            lines[0],
            tranformed_origin,
            tranformed_x_pos,
            rgbColor=RED_RGB,
        )
        move_line(
            lines[1],
            tranformed_origin,
            tranformed_y_pos,
            rgbColor=GREEN_RGB,
        )
        move_line(
            lines[2],
            tranformed_origin,
            tranformed_z_pos,
            rgbColor=BLUE_RGB,
        )
        return lines


class VisualizationNode:
    def __init__(self):
        # ROS setup
        rospy.init_node("visualization_ros_node")

        # ROS msgs
        self.iiwa_joint_cmd = None
        self.allegro_joint_cmd = None
        self.iiwa_joint_state = None
        self.allegro_joint_state = None
        self.palm_target = None

        # Subscribers
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
        self.palm_target_sub = rospy.Subscriber(
            "/palm_target", Float64MultiArray, self.palm_target_callback
        )

        # Initialize PyBullet
        # Create a real robot (simulating real robot) and a command robot (visualizing commands)
        rospy.loginfo("~" * 80)
        rospy.loginfo("Initializing PyBullet")
        self.initialize_pybullet()
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
            p.changeVisualShape(
                robot_cmd_id, link_index, rgbaColor=BLUE_TRANSLUCENT_RGBA
            )

        # Set the robot to a default pose
        DEFAULT_ARM_Q = np.zeros(NUM_ARM_JOINTS)
        DEFAULT_HAND_Q = np.zeros(NUM_HAND_JOINTS)
        assert DEFAULT_ARM_Q.shape == (NUM_ARM_JOINTS,)
        assert DEFAULT_HAND_Q.shape == (NUM_HAND_JOINTS,)
        DEFAULT_Q = np.concatenate([DEFAULT_ARM_Q, DEFAULT_HAND_Q])
        self.set_robot_state(robot_id, DEFAULT_Q)
        self.set_robot_state(robot_cmd_id, DEFAULT_Q)

        # Keep track of the link names and IDs
        self.robot_link_name_to_id = {}
        for i in range(p.getNumJoints(robot_id)):
            self.robot_link_name_to_id[
                p.getJointInfo(robot_id, i)[12].decode("utf-8")
            ] = i
        self.robot_cmd_link_name_to_id = {}
        for i in range(p.getNumJoints(robot_cmd_id)):
            self.robot_cmd_link_name_to_id[
                p.getJointInfo(robot_cmd_id, i)[12].decode("utf-8")
            ] = i

        # Create the hand target
        FAR_AWAY_PALM_TARGET = np.zeros(6) + 100  # Far away
        self.hand_target_lines = visualize_transform(
            xyz=FAR_AWAY_PALM_TARGET[:3],
            rotation_matrix=R.from_euler("zyx", FAR_AWAY_PALM_TARGET[3:]).as_matrix(),
        )
        self.hand_lines = visualize_transform(
            xyz=FAR_AWAY_PALM_TARGET[:3],
            rotation_matrix=R.from_euler("zyx", FAR_AWAY_PALM_TARGET[3:]).as_matrix(),
        )
        self.hand_cmd_lines = visualize_transform(
            xyz=FAR_AWAY_PALM_TARGET[:3],
            rotation_matrix=R.from_euler("zyx", FAR_AWAY_PALM_TARGET[3:]).as_matrix(),
        )

        # Set the camera parameters
        self.set_pybullet_camera()

        # Set gravity for simulation
        p.setGravity(0, 0, -9.81)

        # Draw the world
        world_dict = world_dict_robot_frame
        for object_name, object_dict in world_dict.items():
            xyz_qxyzw = np.array([float(x) for x in object_dict["transform"].split()])
            scaling = np.array([float(x) for x in object_dict["scaling"].split()])
            assert len(xyz_qxyzw) == 7, f"xyz_qxyzw: {xyz_qxyzw}"
            assert len(scaling) == 3, f"scaling: {scaling}"

            half_extents = [x / 2 for x in scaling]
            object_pos = xyz_qxyzw[:3]
            object_quat_xyzw = xyz_qxyzw[3:]
            add_cuboid(
                halfExtents=half_extents,
                position=object_pos,
                orientation=object_quat_xyzw,
            )

        self.robot_id, self.robot_cmd_id = robot_id, robot_cmd_id

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

    def palm_target_callback(self, msg: Float64MultiArray):
        """Callback to update the current hand target."""
        self.palm_target = np.array(msg.data)

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

        if self.palm_target is None:
            rospy.logwarn("palm_target is None")
            palm_target = np.zeros(6) + 100  # Far away
        else:
            palm_target = self.palm_target

        # Command Robot: Set the commanded joint positions
        q_cmd = np.concatenate([iiwa_joint_cmd, allegro_joint_cmd])
        q_state = np.concatenate([iiwa_joint_state, allegro_joint_state])
        self.set_robot_state(self.robot_cmd_id, q_cmd)
        self.set_robot_state(self.robot_id, q_state)

        # Update the hand target
        visualize_transform(
            xyz=palm_target[:3],
            rotation_matrix=R.from_euler("zyx", palm_target[3:]).as_matrix(),
            lines=self.hand_target_lines,
        )

        # Visualize the palm of the robot
        robot_palm_com, robot_palm_quat, *_ = p.getLinkState(
            self.robot_id,
            self.robot_link_name_to_id["palm_link"],
            computeForwardKinematics=1,
        )
        robot_cmd_palm_com, robot_cmd_palm_quat, *_ = p.getLinkState(
            self.robot_cmd_id,
            self.robot_cmd_link_name_to_id["palm_link"],
            computeForwardKinematics=1,
        )
        visualize_transform(
            xyz=np.array(robot_palm_com),
            rotation_matrix=R.from_quat(robot_palm_quat).as_matrix(),
            lines=self.hand_lines,
        )
        visualize_transform(
            xyz=np.array(robot_cmd_palm_com),
            rotation_matrix=R.from_quat(robot_cmd_palm_quat).as_matrix(),
            lines=self.hand_cmd_lines,
        )

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
