#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import torch
from vec_olivia_reference import (
    VecOliviaReferenceMotion,
)


class GoalObjectPosePublisher:
    def __init__(self):
        rospy.init_node("goal_object_pose_publisher")

        self.pose_pub = rospy.Publisher("/goal_object_pose", Pose, queue_size=1)
        self.rate = rospy.Rate(60)  # 60Hz

        # Set up VecOliviaReferenceMotion
        TRAJECTORY_FOLDERPATH = Path(
            "/juno/u/oliviayl/repos/cross_embodiment/FP_outputs/0/"
        )
        HAND_TRAJECTORY_FOLDERPATH = Path(
            "/juno/u/oliviayl/repos/cross_embodiment/hamer/outputs4/0/"
        )

        self.trajectory = VecOliviaReferenceMotion(
            trajectory_folder=TRAJECTORY_FOLDERPATH,
            hand_folder=HAND_TRAJECTORY_FOLDERPATH,
            batch_size=1,
            device="cuda",
            dt=1 / 30,
        )

        self.T_C_O_list = self.extract_object_poses()
        self.current_index = 0
        self.N = len(self.T_C_O_list)

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
        from torch_jit_utils import (
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


if __name__ == "__main__":
    try:
        node = GoalObjectPosePublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
