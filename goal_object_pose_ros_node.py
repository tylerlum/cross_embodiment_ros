#!/usr/bin/env python

from pathlib import Path
from typing import Literal
import functools

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Pose
from isaacgymenvs.utils.cross_embodiment.camera_extrinsics import (
    ZED_CAMERA_T_R_C,
    REALSENSE_CAMERA_T_R_C,
)
from isaacgymenvs.utils.cross_embodiment.vec_olivia_reference import (
    VecOliviaReferenceMotion,
)
from scipy.spatial.transform import Rotation as R


class GoalObjectPosePublisher:
    def __init__(self):
        rospy.init_node("goal_object_pose_publisher")

        self.pose_pub = rospy.Publisher("/goal_object_pose", Pose, queue_size=1)
        self.rate_hz = 60
        self.rate = rospy.Rate(self.rate_hz)

        self.current_index = 0

        MODE: Literal["trajectory", "position"] = "trajectory"
        if MODE == "trajectory":
            TRAJECTORY_INDEX = 8
            # OBJECT_NAME = "snackbox"
            OBJECT_NAME = "bluecup_tape_top"

            # Set up VecOliviaReferenceMotion
            TRAJECTORY_FOLDERPATH = Path(
                f"/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/debug_archive/{OBJECT_NAME}/{TRAJECTORY_INDEX}"
            )
            HAND_TRAJECTORY_FOLDERPATH = Path(
                f"/juno/u/oliviayl/repos/cross_embodiment/hamer/outputs/{OBJECT_NAME}/{TRAJECTORY_INDEX}"
            )

            self.data_hz = 30
            self.trajectory = VecOliviaReferenceMotion(
                trajectory_folder=TRAJECTORY_FOLDERPATH,
                hand_folder=HAND_TRAJECTORY_FOLDERPATH,
                batch_size=1,
                device="cuda",
                dt=1 / self.data_hz,
            )

            self.T_C_O_list = self.extract_object_poses()
            self.N_STEPS_PER_UPDATE = 4

            # Extend list
            new_list = []
            for T_C_O in self.T_C_O_list:
                new_list += [T_C_O] * self.N_STEPS_PER_UPDATE
            self.T_C_O_list = new_list

            self.N = len(self.T_C_O_list)

        elif MODE == "position":
            # goal_object_pos = np.array([0.4637, -0.2200, 0.5199])
            goal_object_pos = np.array([0.5735, -0.1633, 0.2038]) + np.array(
                [0.0, -0.12, 0.35]
            )
            goal_object_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
            T_R_O = np.eye(4)
            T_R_O[:3, :3] = R.from_quat(goal_object_quat_xyzw).as_matrix()
            T_R_O[:3, 3] = goal_object_pos

            T_C_R = np.linalg.inv(self.goal_T_R_C)
            T_C_O = T_C_R @ T_R_O

            self.T_C_O_list = [T_C_O]
            self.N = len(self.T_C_O_list)
        else:
            raise ValueError(f"Invalid mode {MODE}")

    def extract_object_poses(self):
        T_C_O_list = []
        while not self.trajectory.done:
            goal_object_pos, goal_object_quat_xyzw, _, _ = (
                self._get_goal_object_pose_olivia_helper(
                    reference_motion=self.trajectory,
                    device=self.trajectory._device,
                    num_envs=self.trajectory._batch_size,
                )
            )
            T_C_O_list.append(
                self.create_transform(
                    pos=goal_object_pos[0].cpu().numpy(),
                    rot=self.quat_xyzw_to_matrix(
                        goal_object_quat_xyzw[0].cpu().numpy()
                    ),
                )
            )
            self.trajectory.step_idx(
                step_env_ids=torch.ones(
                    1,
                )
                .nonzero(as_tuple=False)
                .squeeze(dim=-1)
            )
        return T_C_O_list

    def _get_goal_object_pose_olivia_helper(self, reference_motion, device, num_envs):
        from isaacgymenvs.utils.torch_jit_utils import (
            matrix_to_quat_xyzw,
            quat_xyzw_to_matrix,
        )

        T_C_Os = (
            torch.eye(4, device=device)
            .unsqueeze(dim=0)
            .repeat_interleave(num_envs, dim=0)
        )
        T_C_Os[:, :3, 3] = reference_motion.object_pos
        T_C_Os[:, :3, :3] = quat_xyzw_to_matrix(reference_motion.object_quat_xyzw)

        new_goal_object_pos = T_C_Os[:, :3, 3]
        new_goal_object_quat_xyzw = matrix_to_quat_xyzw(T_C_Os[:, :3, :3])

        return new_goal_object_pos, new_goal_object_quat_xyzw, None, None

    @staticmethod
    def create_transform(pos, rot):
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T

    @staticmethod
    def quat_xyzw_to_matrix(quat_xyzw):
        return R.from_quat(quat_xyzw).as_matrix()

    def publish_pose(self):
        if self.current_index >= self.N:
            self.current_index = self.N - 1

        T = self.T_C_O_list[self.current_index]
        trans = T[:3, 3]
        quat_xyzw = R.from_matrix(T[:3, :3]).as_quat()

        msg = Pose()
        msg.position.x, msg.position.y, msg.position.z = trans
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = (
            quat_xyzw
        )

        self.pose_pub.publish(msg)
        rospy.logdebug(f"Pose {self.current_index} published to /goal_object_pose")

        self.current_index += 1

    def run(self):
        rospy.loginfo("Publishing goal object poses at 60Hz to /goal_object_pose")
        while not rospy.is_shutdown():
            self.publish_pose()
            self.rate.sleep()

    @property
    @functools.lru_cache()
    def goal_T_R_C(self) -> np.ndarray:
        # Check camera parameter
        camera = rospy.get_param("/goal_camera", None)
        if camera is None:
            DEFAULT_CAMERA = "zed"
            rospy.logwarn(
                f"No /camera parameter found, using default camera {DEFAULT_CAMERA}"
            )
            camera = DEFAULT_CAMERA
        rospy.loginfo(f"Using camera: {camera}")
        if camera == "zed":
            return ZED_CAMERA_T_R_C
        elif camera == "realsense":
            return REALSENSE_CAMERA_T_R_C
        else:
            raise ValueError(f"Unknown camera: {camera}")


if __name__ == "__main__":
    try:
        node = GoalObjectPosePublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
