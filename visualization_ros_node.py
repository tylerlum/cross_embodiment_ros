#!/usr/bin/env python

import trimesh
import struct
import time
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pybullet as p
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import Float64MultiArray

from fabric_world import world_dict_robot_frame
from camera_extrinsics import T_R_C, T_R_C2

NUM_ARM_JOINTS = 7
NUM_HAND_JOINTS = 16
BLUE_TRANSLUCENT_RGBA = [0, 0, 1, 0.5]
RED_TRANSLUCENT_RGBA = [1, 0, 0, 0.2]
GREEN_TRANSLUCENT_RGBA = [0, 1, 0, 0.5]

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


def rgb_to_float(color: float) -> Tuple[float, float, float]:
    """Convert packed RGB float to separate R, G, B components (https://wiki.ros.org/pcl/Overview)."""
    s = struct.pack(">f", color)
    i = struct.unpack(">I", s)[0]
    r = (i >> 16) & 0x0000FF
    g = (i >> 8) & 0x0000FF
    b = (i) & 0x0000FF
    return r / 255.0, g / 255.0, b / 255.0


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    assert T.shape == (4, 4), T.shape
    n_pts = points.shape[0]
    assert points.shape == (n_pts, 3), points.shape

    return (T[:3, :3] @ points.T + T[:3, 3][:, None]).T


def draw_colored_point_cloud(
    point_cloud_and_colors: np.ndarray,
    point_size: int = 5,
):
    n_pts = point_cloud_and_colors.shape[0]
    assert point_cloud_and_colors.shape == (
        n_pts,
        4,
    ), f"point_cloud_and_colors.shape: {point_cloud_and_colors.shape}"
    point_cloud = point_cloud_and_colors[:, :3]

    # point_cloud is in C2 frame
    point_cloud_R = transform_points(T=T_R_C2, points=point_cloud)

    # Extract colors from the point cloud
    point_cloud_raw_colors = point_cloud_and_colors[:, 3]
    point_cloud_colors = [rgb_to_float(color) for color in point_cloud_raw_colors]

    # Use debug points instead of spheres for faster rendering
    if not hasattr(draw_colored_point_cloud, "points"):
        draw_colored_point_cloud.points = p.addUserDebugPoints(
            point_cloud_R,
            point_cloud_colors,
            point_size,
        )
    else:
        p.addUserDebugPoints(
            point_cloud_R,
            point_cloud_colors,
            point_size,
            replaceItemUniqueId=draw_colored_point_cloud.points,
        )


def create_urdf(obj_path: Path) -> Path:
    assert obj_path.suffix == ".obj"
    filename = obj_path.name
    parent_folder = obj_path.parent
    urdf_path = parent_folder / f"{obj_path.stem}.urdf"
    urdf_text = f"""<?xml version="1.0" ?>
<robot name="model.urdf">
  <link name="baseLink">
    <contact>
      <lateral_friction value="0.8"/>
      <rolling_friction value="0.001"/>
      <contact_cfm value="0.0"/>
      <contact_erp value="1.0"/>
    </contact>
    <inertial>
    <origin rpy="0 0 0" xyz="0.01 0.0 0.01"/>
       <mass value=".066"/>
       <inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="{filename}" scale="1 1 1"/>
      </geometry>
      <material name="white">
        <color rgba="1. 1. 1. 1."/>
      </material>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{filename}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>"""
    with urdf_path.open("w") as f:
        f.write(urdf_text)
    return urdf_path


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
        self.object_pose = None
        self.goal_object_pose = None
        self.point_cloud_and_colors = None

        # Subscribers
        self.iiwa_sub = rospy.Subscriber(
            "/iiwa/joint_states", JointState, self.iiwa_joint_state_callback
        )
        self.allegro_sub = rospy.Subscriber(
            "/allegroHand_0/joint_states", JointState, self.allegro_joint_state_callback
        )
        self.iiwa_cmd_sub = rospy.Subscriber(
            "/iiwa/joint_cmd", JointState, self.iiwa_joint_cmd_callback
        )
        self.allegro_cmd_sub = rospy.Subscriber(
            "/allegroHand_0/joint_cmd", JointState, self.allegro_joint_cmd_callback
        )
        self.palm_target_sub = rospy.Subscriber(
            "/palm_target", Float64MultiArray, self.palm_target_callback
        )
        self.object_pose_sub = rospy.Subscriber(
            "/object_pose", Pose, self.object_pose_callback
        )
        self.goal_object_pose_sub = rospy.Subscriber(
            "/goal_object_pose", Pose, self.goal_object_pose_callback
        )
        self.point_cloud_sub = rospy.Subscriber(
            "/zed/zed_node/point_cloud/cloud_registered",
            PointCloud2,
            self.point_cloud_callback,
        )

        # Initialize PyBullet
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

        # Create a real robot (simulating real robot) and a command robot (visualizing commands)
        # Load robot URDF with a fixed base
        robot_urdf_path = Path(
            # "/juno/u/tylerlum/github_repos/bidexhands_isaacgymenvs/assets/urdf/kuka_allegro_description/kuka_allegro.urdf"
            "/juno/u/tylerlum/github_repos/fabrics-sim/src/fabrics_sim/models/robots/urdf/kuka_allegro/kuka_allegro.urdf"
        )
        assert robot_urdf_path.exists(), f"robot_urdf_path not found: {robot_urdf_path}"
        self.robot_id = p.loadURDF(str(robot_urdf_path), useFixedBase=True)
        self.robot_cmd_id = p.loadURDF(str(robot_urdf_path), useFixedBase=True)

        # Load the scene mesh
        scene_urdf_path = create_urdf(
            Path("/juno/u/tylerlum/Downloads/kuka_table/new_origin_9/kuka_table.obj")
        )
        _scene_id = p.loadURDF(str(scene_urdf_path), useFixedBase=True)

        # Load the object mesh
        FAR_AWAY_OBJECT_POSITION = np.ones(3)
        object_mesh_path = rospy.get_param("/mesh_file", None)
        if object_mesh_path is None:
            DEFAULT_MESH_PATH = (
                # "/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/kiri_meshes/blueblock/3DModel.obj"
                # "/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/kiri_meshes/snackbox/3DModel.obj"
                # "/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/kiri_meshes/woodblock/3DModel.obj"
                # "/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/kiri_meshes/cup_ycbv/textured.obj"
                "/juno/u/tylerlum/github_repos/cross_embodiment_ros/kiri_meshes/cup_ycbv/textured.obj"
            )
            object_mesh_path = DEFAULT_MESH_PATH
            rospy.logwarn(f"Using default object mesh: {object_mesh_path}")
        assert isinstance(
            object_mesh_path, str
        ), f"object_mesh_path: {object_mesh_path}"

        # Object often has too many faces and may have issues loading
        MAX_N_FACES = 10_000
        if trimesh.load(object_mesh_path).faces.shape[0] > MAX_N_FACES:
            rospy.logwarn(
                f"object_mesh_path: {object_mesh_path} has more than {MAX_N_FACES} faces"
            )

            import open3d as o3d

            orig_mesh = o3d.io.read_triangle_mesh(object_mesh_path)
            simplified_mesh = orig_mesh.simplify_quadric_decimation(
                target_number_of_triangles=MAX_N_FACES,
            )
            object_mesh_Path = Path(object_mesh_path)
            simplified_mesh_path = str(
                object_mesh_Path.parent
                / f"{object_mesh_Path.stem}_simplified{object_mesh_Path.suffix}"
            )
            o3d.io.write_triangle_mesh(simplified_mesh_path, simplified_mesh)
            rospy.logwarn(f"Saved simplified mesh to: {simplified_mesh_path}")
            object_mesh_path = simplified_mesh_path

        object_urdf_path = create_urdf(Path(object_mesh_path))
        self.object_id = p.loadURDF(str(object_urdf_path), useFixedBase=True)
        p.resetBasePositionAndOrientation(
            self.object_id, FAR_AWAY_OBJECT_POSITION, [0, 0, 0, 1]
        )

        self.goal_object_id = p.loadURDF(str(object_urdf_path), useFixedBase=True)
        p.resetBasePositionAndOrientation(
            self.goal_object_id,
            FAR_AWAY_OBJECT_POSITION + np.array([0.2, 0.2, 0.2]),
            [0, 0, 0, 1],
        )
        p.changeVisualShape(self.goal_object_id, -1, rgbaColor=GREEN_RGBA)

        # Make the robot blue
        # Change the color of each link (including the base)
        num_joints = p.getNumJoints(self.robot_id)
        for link_index in range(-1, num_joints):  # -1 is for the base
            p.changeVisualShape(
                self.robot_cmd_id, link_index, rgbaColor=BLUE_TRANSLUCENT_RGBA
            )

        # Set the robot to a default pose
        DEFAULT_ARM_Q = np.zeros(NUM_ARM_JOINTS)
        DEFAULT_HAND_Q = np.zeros(NUM_HAND_JOINTS)
        assert DEFAULT_ARM_Q.shape == (NUM_ARM_JOINTS,)
        assert DEFAULT_HAND_Q.shape == (NUM_HAND_JOINTS,)
        DEFAULT_Q = np.concatenate([DEFAULT_ARM_Q, DEFAULT_HAND_Q])
        self.set_robot_state(self.robot_id, DEFAULT_Q)
        self.set_robot_state(self.robot_cmd_id, DEFAULT_Q)

        # Keep track of the link names and IDs
        self.robot_link_name_to_id = {}
        for i in range(p.getNumJoints(self.robot_id)):
            self.robot_link_name_to_id[
                p.getJointInfo(self.robot_id, i)[12].decode("utf-8")
            ] = i
        self.robot_cmd_link_name_to_id = {}
        for i in range(p.getNumJoints(self.robot_cmd_id)):
            self.robot_cmd_link_name_to_id[
                p.getJointInfo(self.robot_cmd_id, i)[12].decode("utf-8")
            ] = i

        # Create the hand target
        FAR_AWAY_PALM_TARGET = np.concatenate([np.ones(3), np.zeros(3)])
        self.hand_target_lines = visualize_transform(
            xyz=FAR_AWAY_PALM_TARGET[:3],
            rotation_matrix=R.from_euler("ZYX", FAR_AWAY_PALM_TARGET[3:]).as_matrix(),
        )
        self.hand_lines = visualize_transform(
            xyz=FAR_AWAY_PALM_TARGET[:3],
            rotation_matrix=R.from_euler("ZYX", FAR_AWAY_PALM_TARGET[3:]).as_matrix(),
        )
        self.hand_cmd_lines = visualize_transform(
            xyz=FAR_AWAY_PALM_TARGET[:3],
            rotation_matrix=R.from_euler("ZYX", FAR_AWAY_PALM_TARGET[3:]).as_matrix(),
        )

        # Create the camera lines
        CAMERA_LINES_TO_DRAW: Literal["C", "C2"] = "C"
        if CAMERA_LINES_TO_DRAW == "C":
            self.camera_lines = visualize_transform(
                xyz=T_R_C[:3, 3],
                rotation_matrix=T_R_C[:3, :3],
            )
        elif CAMERA_LINES_TO_DRAW == "C2":
            self.camera_lines = visualize_transform(
                xyz=T_R_C2[:3, 3],
                rotation_matrix=T_R_C2[:3, :3],
            )
        else:
            raise ValueError(f"Invalid CAMERA_LINES_TO_DRAW: {CAMERA_LINES_TO_DRAW}")

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

    def object_pose_callback(self, msg: Pose):
        """ "Callback to update the current object pose."""
        xyz = np.array([msg.position.x, msg.position.y, msg.position.z])
        quat_xyzw = np.array(
            [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ]
        )
        latest_pose = np.eye(4)
        latest_pose[:3, 3] = xyz
        latest_pose[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        self.object_pose = latest_pose

    def goal_object_pose_callback(self, msg: Pose):
        """ "Callback to update the goal object pose."""
        xyz = np.array([msg.position.x, msg.position.y, msg.position.z])
        quat_xyzw = np.array(
            [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ]
        )
        latest_pose = np.eye(4)
        latest_pose[:3, 3] = xyz
        latest_pose[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        self.goal_object_pose = latest_pose

    def point_cloud_callback(self, msg: PointCloud2):
        self.point_cloud_and_colors = np.array(
            list(
                pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
            )
        )

        n_pts = self.point_cloud_and_colors.shape[0]
        assert self.point_cloud_and_colors.shape == (
            n_pts,
            4,
        ), f"self.point_cloud_and_colors.shape: {self.point_cloud_and_colors.shape}"

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

        if self.object_pose is None:
            rospy.logwarn("object_pose is None")
            object_pose = np.eye(4)
            object_pose[:3, 3] = np.zeros(3) + 100  # Far away
        else:
            object_pose = self.object_pose

        if self.goal_object_pose is None:
            rospy.logwarn("goal_object_pose is None")
            goal_object_pose = np.eye(4)
            goal_object_pose[:3, 3] = np.zeros(3) + 100  # Far away
        else:
            goal_object_pose = self.goal_object_pose

        # Command Robot: Set the commanded joint positions
        q_cmd = np.concatenate([iiwa_joint_cmd, allegro_joint_cmd])
        q_state = np.concatenate([iiwa_joint_state, allegro_joint_state])
        self.set_robot_state(self.robot_cmd_id, q_cmd)
        self.set_robot_state(self.robot_id, q_state)

        # Update the hand target
        visualize_transform(
            xyz=palm_target[:3],
            rotation_matrix=R.from_euler("ZYX", palm_target[3:]).as_matrix(),
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

        # Log to debug palm position and orientation in robot frame
        rospy.loginfo(f"robot_palm_com = {robot_palm_com}")
        rospy.loginfo(f"robot_palm_quat = {robot_palm_quat}")
        robot_palm_euler_ZYX = R.from_quat(robot_palm_quat).as_euler(
            "ZYX", degrees=False
        )
        rospy.loginfo(f"robot_palm_euler_ZYX = {robot_palm_euler_ZYX}")

        # Update the object pose
        # Object pose is in camera frame = C frame
        # We want it in world frame = robot frame = R frame
        T_C_O = object_pose
        T_R_O = T_R_C @ T_C_O
        object_pos = T_R_O[:3, 3]
        object_quat_xyzw = R.from_matrix(T_R_O[:3, :3]).as_quat()
        p.resetBasePositionAndOrientation(self.object_id, object_pos, object_quat_xyzw)

        # Update the goal object pose
        # Goal object pose is in camera frame = C frame
        # We want it in world frame = robot frame = R frame
        T_C_G = goal_object_pose
        T_R_G = T_R_C @ T_C_G
        goal_object_pos = T_R_G[:3, 3]
        goal_object_quat_xyzw = R.from_matrix(T_R_G[:3, :3]).as_quat()
        p.resetBasePositionAndOrientation(
            self.goal_object_id, goal_object_pos, goal_object_quat_xyzw
        )

        # Update the point cloud
        if self.point_cloud_and_colors is not None:
            draw_colored_point_cloud(
                point_cloud_and_colors=self.point_cloud_and_colors,
            )
        else:
            rospy.logwarn("point_cloud_and_colors is None")

    def run(self):
        """Main loop to run the node, update simulation, and publish joint states."""
        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            # Update the PyBullet simulation with the current joint commands
            self.update_pybullet()

            # Sleep to maintain the loop rate
            before_sleep_time = rospy.Time.now()

            SLEEP_MODE: Literal[
                "ROS_SLEEP", "WHILE_SLEEP", "WHILE_NO_SLEEP", "NO_SLEEP"
            ] = "NO_SLEEP"
            if SLEEP_MODE == "ROS_SLEEP":
                # Seems to cause segfault?
                self.rate.sleep()
            elif SLEEP_MODE == "WHILE_SLEEP":
                loops_waiting = 0
                while (
                    not rospy.is_shutdown()
                    and (rospy.Time.now() - start_time).to_sec() < 1 / self.rate_hz
                ):
                    rospy.loginfo(
                        f"loops_waiting: {loops_waiting}, (rospy.Time.now() - start_time).to_sec() = {(rospy.Time.now() - start_time).to_sec()}"
                    )
                    loops_waiting += 1
                    time.sleep(0.0001)
            elif SLEEP_MODE == "WHILE_NO_SLEEP":
                loops_waiting = 0
                while (
                    not rospy.is_shutdown()
                    and (rospy.Time.now() - start_time).to_sec() < 1 / self.rate_hz
                ):
                    rospy.loginfo(
                        f"loops_waiting: {loops_waiting}, (rospy.Time.now() - start_time).to_sec() = {(rospy.Time.now() - start_time).to_sec()}"
                    )
                    loops_waiting += 1
            elif SLEEP_MODE == "NO_SLEEP":
                pass
            else:
                raise ValueError(f"Invalid SLEEP_MODE: {SLEEP_MODE}")

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
