#!/usr/bin/env python

from typing import Optional, Tuple
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
import torch
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout

from rl_player import RlPlayer
from camera_extrinsics import T_R_C


def var_to_is_none_str(var) -> str:
    if var is None:
        return "None"
    return "Not None"


def assert_equals(a, b) -> None:
    assert a == b, f"{a} != {b}"


NUM_XYZ = 3
NUM_ARM_DOFS = 7
NUM_DOFS_PER_FINGER = 4
NUM_FINGERS = 4
NUM_HAND_DOFS = NUM_DOFS_PER_FINGER * NUM_FINGERS
NUM_HAND_ARM_DOFS = NUM_ARM_DOFS + NUM_HAND_DOFS
KUKA_ALLEGRO_NUM_DOFS = NUM_HAND_ARM_DOFS
KUKA_ALLEGRO_ASSET_ROOT = "/juno/u/tylerlum/github_repos/fabrics-sim/src/fabrics_sim/models/robots/urdf/kuka_allegro"
KUKA_ALLEGRO_FILENAME = "kuka_allegro.urdf"
PALM_LINK_NAME = "palm_link"
PALM_X_LINK_NAME = "palm_x"
PALM_Y_LINK_NAME = "palm_y"
PALM_Z_LINK_NAME = "palm_z"
PALM_LINK_NAMES = [PALM_LINK_NAME, PALM_X_LINK_NAME, PALM_Y_LINK_NAME, PALM_Z_LINK_NAME]
ALLEGRO_FINGERTIP_LINK_NAMES = [
    "index_biotac_tip",
    "middle_biotac_tip",
    "ring_biotac_tip",
    "thumb_biotac_tip",
]


class RLPolicyNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node("rl_policy_node")

        # Publisher for palm and hand targets
        self.palm_target_pub = rospy.Publisher(
            "/palm_target", Float64MultiArray, queue_size=10
        )
        self.hand_target_pub = rospy.Publisher(
            "/hand_target", Float64MultiArray, queue_size=10
        )

        # Variables to store the latest messages
        self.object_pose_msg = None
        self.goal_object_pose_msg = None
        self.iiwa_joint_state_msg = None
        self.allegro_joint_state_msg = None
        self.fabric_state_msg = None

        # Subscribers
        self.object_pose_sub = rospy.Subscriber(
            "/object_pose", Pose, self.object_pose_callback
        )
        self.goal_object_pose_sub = rospy.Subscriber(
            "/goal_object_pose", Pose, self.goal_object_pose_callback
        )
        self.iiwa_joint_state_sub = rospy.Subscriber(
            "/iiwa/joint_states", JointState, self.iiwa_joint_state_callback
        )
        self.allegro_joint_state_sub = rospy.Subscriber(
            "/allegroHand_0/joint_states", JointState, self.allegro_joint_state_callback
        )
        self.fabric_sub = rospy.Subscriber(
            "/fabric_state", JointState, self.fabric_state_callback
        )

        # ROS rate (60Hz)
        self.rate = rospy.Rate(60)  # 60 Hz

        # RL Player setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_observations = 130  # Update this number based on actual dimensions
        self.num_actions = 11  # First 6 for palm, last 5 for hand
        self.config_path = "/move/u/tylerlum/github_repos/bidexhands_isaacgymenvs/isaacgymenvs/runs/RIGHT_1-freq_coll-on_damp-25_move3_2024-10-02_04-42-29-349841/config_resolved.yaml"  # Update this path
        self.checkpoint_path = "/move/u/tylerlum/github_repos/bidexhands_isaacgymenvs/isaacgymenvs/runs/RIGHT_1-freq_coll-on_damp-25_move3_2024-10-02_04-42-29-349841/nn/last_RIGHT_1-freq_coll-on_damp-25_move3_ep_13000_rew_124.79679.pth"  # Update this path

        # Create the RL player
        self.player = RlPlayer(
            num_observations=self.num_observations,
            num_actions=self.num_actions,
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

        # Define limits for palm and hand targets
        self.palm_mins = torch.tensor(
            [1.0, -0.7, 0, -3.1416, -3.1416, -3.1416], device=self.device
        )
        self.palm_maxs = torch.tensor(
            [0.0, 0.7, 1.0, 3.1416, 3.1416, 3.1416], device=self.device
        )
        self.hand_mins = torch.tensor(
            [0.2475, -0.3286, -0.7238, -0.0192, -0.5532], device=self.device
        )
        self.hand_maxs = torch.tensor(
            [3.8336, 3.0025, 0.8977, 1.0243, 0.0629], device=self.device
        )

        self._setup_taskmap()

    def object_pose_callback(self, msg: Pose):
        self.object_pose_msg = msg

    def goal_object_pose_callback(self, msg: Pose):
        self.goal_object_pose_msg = msg

    def iiwa_joint_state_callback(self, msg: JointState):
        self.iiwa_joint_state_msg = msg

    def allegro_joint_state_callback(self, msg: JointState):
        self.allegro_joint_state_msg = msg

    def fabric_state_callback(self, msg: JointState):
        self.fabric_state_msg = msg

    def create_observation(self) -> Optional[torch.Tensor]:
        # Ensure all messages are received before processing
        if (
            self.iiwa_joint_state_msg is None
            or self.allegro_joint_state_msg is None
            or self.object_pose_msg is None
            or self.goal_object_pose_msg is None
            or self.fabric_state_msg is None
        ):
            rospy.logwarn(
                f"Waiting for all messages to be received... iiwa_joint_state_msg: {var_to_is_none_str(self.iiwa_joint_state_msg)}, allegro_joint_state_msg: {var_to_is_none_str(self.allegro_joint_state_msg)}, object_pose_msg: {var_to_is_none_str(self.object_pose_msg)}, goal_object_pose_msg: {var_to_is_none_str(self.goal_object_pose_msg)}, fabric_state_msg: {var_to_is_none_str(self.fabric_state_msg)}"
            )
            return None

        # Concatenate the data from joint states and object pose
        iiwa_position = np.array(self.iiwa_joint_state_msg.position)
        iiwa_velocity = np.array(self.iiwa_joint_state_msg.velocity)

        allegro_position = np.array(self.allegro_joint_state_msg.position)
        allegro_velocity = np.array(self.allegro_joint_state_msg.velocity)

        object_position_C = np.array(
            [
                self.object_pose_msg.position.x,
                self.object_pose_msg.position.y,
                self.object_pose_msg.position.z,
            ]
        )
        object_quat_xyzw_C = np.array(
            [
                self.object_pose_msg.orientation.x,
                self.object_pose_msg.orientation.y,
                self.object_pose_msg.orientation.z,
                self.object_pose_msg.orientation.w,
            ]
        )
        T_C_O = np.eye(4)
        T_C_O[:3, 3] = object_position_C
        T_C_O[:3, :3] = R.from_quat(object_quat_xyzw_C).as_matrix()

        goal_object_pos_C = np.array(
            [
                self.goal_object_pose_msg.position.x,
                self.goal_object_pose_msg.position.y,
                self.goal_object_pose_msg.position.z,
            ]
        )
        goal_object_quat_xyzw_C = np.array(
            [
                self.goal_object_pose_msg.orientation.x,
                self.goal_object_pose_msg.orientation.y,
                self.goal_object_pose_msg.orientation.z,
                self.goal_object_pose_msg.orientation.w,
            ]
        )
        T_C_G = np.eye(4)
        T_C_G[:3, 3] = goal_object_pos_C
        T_C_G[:3, :3] = R.from_quat(goal_object_quat_xyzw_C).as_matrix()

        T_R_O = T_R_C @ T_C_O
        object_position_R = T_R_O[:3, 3]
        object_quat_xyzw_R = R.from_matrix(T_R_O[:3, :3]).as_quat()

        T_R_G = T_R_C @ T_C_G
        goal_object_pos_R = T_R_G[:3, 3]
        goal_object_quat_xyzw_R = R.from_matrix(T_R_G[:3, :3]).as_quat()

        q = np.concatenate([iiwa_position, allegro_position])
        qd = np.concatenate([iiwa_velocity, allegro_velocity])

        fabric_q = np.array(self.fabric_state_msg.position)
        fabric_qd = np.array(self.fabric_state_msg.velocity)
        fabric_qdd = np.array(self.fabric_state_msg.effort)

        taskmap_positions, _, _ = self.taskmap_helper(
            q=torch.from_numpy(q).float().unsqueeze(0).to(self.device),
            qd=torch.from_numpy(qd).float().unsqueeze(0).to(self.device),
        )
        taskmap_positions = taskmap_positions.squeeze(0).cpu().numpy()
        palm_pos = taskmap_positions[self.taskmap_link_names.index(PALM_LINK_NAME)]
        palm_x_pos = taskmap_positions[self.taskmap_link_names.index(PALM_X_LINK_NAME)]
        palm_y_pos = taskmap_positions[self.taskmap_link_names.index(PALM_Y_LINK_NAME)]
        palm_z_pos = taskmap_positions[self.taskmap_link_names.index(PALM_Z_LINK_NAME)]
        fingertip_positions = np.stack(
            [
                taskmap_positions[self.taskmap_link_names.index(link_name)]
                for link_name in ALLEGRO_FINGERTIP_LINK_NAMES
            ],
            axis=0,
        )

        obs_dict = {}
        obs_dict["q"] = np.concatenate([iiwa_position, allegro_position])
        obs_dict["qd"] = np.concatenate([iiwa_velocity, allegro_velocity])
        obs_dict["fingertip_positions"] = fingertip_positions.reshape(
            NUM_FINGERS * NUM_XYZ
        )
        obs_dict["palm_pos"] = palm_pos
        obs_dict["palm_x_pos"] = palm_x_pos
        obs_dict["palm_y_pos"] = palm_y_pos
        obs_dict["palm_z_pos"] = palm_z_pos
        obs_dict["object_pos"] = object_position_R
        obs_dict["object_quat_xyzw"] = object_quat_xyzw_R
        obs_dict["goal_pos"] = goal_object_pos_R
        obs_dict["goal_quat_xyzw"] = goal_object_quat_xyzw_R
        obs_dict["fabric_q"] = fabric_q
        obs_dict["fabric_qd"] = fabric_qd

        for k, v in obs_dict.items():
            assert len(v.shape) == 1, f"Shape of {k} is {v.shape}, expected 1D tensor"

        # Concatenate all observations into a 1D tensor
        observation = np.concatenate(
            [obs for obs in obs_dict.values()],
            axis=-1,
        )
        assert_equals(observation.shape, (self.num_observations,))

        return torch.from_numpy(observation).float().unsqueeze(0).to(self.device)

    def _setup_taskmap(self) -> None:
        import warp as wp
        wp.init()
        from fabrics_sim.taskmaps.robot_frame_origins_taskmap import (
            RobotFrameOriginsTaskMap,
        )

        # Create task map that consists of the origins of the following frames stacked together.
        self.taskmap_link_names = PALM_LINK_NAMES + ALLEGRO_FINGERTIP_LINK_NAMES
        self.taskmap = RobotFrameOriginsTaskMap(
            urdf_path=str(Path(KUKA_ALLEGRO_ASSET_ROOT) / KUKA_ALLEGRO_FILENAME),
            link_names=self.taskmap_link_names,
            batch_size=1,
            device=self.device,
        )

    def taskmap_helper(
        self, q: torch.Tensor, qd: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = q.shape[0]
        assert_equals(q.shape, (N, KUKA_ALLEGRO_NUM_DOFS))
        assert_equals(qd.shape, (N, KUKA_ALLEGRO_NUM_DOFS))

        x, jac = self.taskmap(q, None)
        n_points = len(self.taskmap_link_names)
        assert_equals(x.shape, (N, NUM_XYZ * n_points))
        assert_equals(jac.shape, (N, NUM_XYZ * n_points, KUKA_ALLEGRO_NUM_DOFS))

        # Calculate the velocity in the task space
        xd = torch.bmm(jac, qd.unsqueeze(2)).squeeze(2)
        assert_equals(xd.shape, (N, NUM_XYZ * n_points))

        return (
            x.reshape(N, n_points, NUM_XYZ),
            xd.reshape(N, n_points, NUM_XYZ),
            jac.reshape(N, n_points, NUM_XYZ, KUKA_ALLEGRO_NUM_DOFS),
        )

    def rescale_action(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Rescale the normalized actions from [-1, 1] to the actual target ranges
        palm_target = (self.palm_maxs - self.palm_mins) * action[:, :6] + self.palm_mins
        hand_target = (self.hand_maxs - self.hand_mins) * action[:, 6:] + self.hand_mins
        return palm_target, hand_target

    def publish_targets(self, palm_target: torch.Tensor, hand_target: torch.Tensor):
        # Convert palm_target to Float64MultiArray and publish
        palm_msg = Float64MultiArray()
        palm_msg.layout = MultiArrayLayout(
            dim=[MultiArrayDimension(label="palm_target", size=6, stride=6)],
            data_offset=0,
        )
        palm_msg.data = palm_target.cpu().numpy().flatten().tolist()
        self.palm_target_pub.publish(palm_msg)

        # Convert hand_target to Float64MultiArray and publish
        hand_msg = Float64MultiArray()
        hand_msg.layout = MultiArrayLayout(
            dim=[MultiArrayDimension(label="hand_target", size=5, stride=5)],
            data_offset=0,
        )
        hand_msg.data = hand_target.cpu().numpy().flatten().tolist()
        self.hand_target_pub.publish(hand_msg)

    def run(self):
        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            # Create observation from the latest messages
            obs = self.create_observation()

            if obs is not None:
                # Get the normalized action from the RL player
                normalized_action = self.player.get_normalized_action(obs=obs)
                # normalized_action = torch.zeros(1, self.num_actions, device=self.device)
                assert_equals(normalized_action.shape, (1, self.num_actions))

                # Rescale the action to get palm and hand targets
                palm_target, hand_target = self.rescale_action(normalized_action)

                # Publish the targets
                self.publish_targets(palm_target, hand_target)

            # Sleep to maintain 60Hz loop rate
            before_sleep_time = rospy.Time.now()
            self.rate.sleep()
            after_sleep_time = rospy.Time.now()
            rospy.loginfo(
                f"Max rate: {1 / (before_sleep_time - start_time).to_sec()} Hz ({(before_sleep_time - start_time).to_sec() * 1000}ms), Actual rate: {1 / (after_sleep_time - start_time).to_sec()} Hz"
            )


if __name__ == "__main__":
    try:
        rl_policy_node = RLPolicyNode()
        rl_policy_node.run()
    except rospy.ROSInterruptException:
        pass
