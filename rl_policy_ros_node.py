#!/usr/bin/env python

import copy
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Pose
from isaacgymenvs.utils.cross_embodiment.camera_extrinsics import T_R_C
from isaacgymenvs.utils.cross_embodiment.constants import (
    NUM_XYZ,
)
from isaacgymenvs.utils.cross_embodiment.kuka_allegro_constants import (
    ALLEGRO_FINGERTIP_LINK_NAMES,
    KUKA_ALLEGRO_ASSET_ROOT,
    KUKA_ALLEGRO_FILENAME,
    NUM_FINGERS,
    PALM_LINK_NAME,
    PALM_LINK_NAMES,
    PALM_X_LINK_NAME,
    PALM_Y_LINK_NAME,
    PALM_Z_LINK_NAME,
)
from isaacgymenvs.utils.cross_embodiment.kuka_allegro_constants import (
    NUM_HAND_ARM_DOFS as KUKA_ALLEGRO_NUM_DOFS,
)
from isaacgymenvs.utils.cross_embodiment.object_constants import (
    NUM_OBJECT_KEYPOINTS,
    OBJECT_KEYPOINT_OFFSETS,
)
from isaacgymenvs.utils.cross_embodiment.utils import (
    assert_equals,
    rescale,
)
from isaacgym.torch_utils import to_torch
from isaacgymenvs.utils.wandb_utils import restore_file_from_wandb
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout

from rl_player import RlPlayer


def var_to_is_none_str(var) -> str:
    if var is None:
        return "None"
    return "Not None"


def pose_msg_to_T(msg: Pose) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.array([msg.position.x, msg.position.y, msg.position.z])
    T[:3, :3] = R.from_quat(
        [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
    ).as_matrix()
    return T


def T_to_pos_quat_xyzw(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pos = T[:3, 3]
    quat_xyzw = R.from_matrix(T[:3, :3]).as_quat()
    return pos, quat_xyzw


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
        self.received_fabric_state_time = None

        self.prev_object_pose_msg = None
        self.prev_prev_object_pose_msg = None

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

        # ROS rate
        self.rate_hz = 15
        self.rate = rospy.Rate(self.rate_hz)

        # RL Player setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_observations = 144  # Update this number based on actual dimensions
        self.num_actions = 11  # First 6 for palm, last 5 for hand
        # self.config_path = "/move/u/tylerlum/github_repos/bidexhands_isaacgymenvs/isaacgymenvs/runs/RIGHT_1-freq_coll-on_damp-25_move3_2024-10-02_04-42-29-349841/config_resolved.yaml"  # Update this path
        # self.checkpoint_path = "/move/u/tylerlum/github_repos/bidexhands_isaacgymenvs/isaacgymenvs/runs/RIGHT_1-freq_coll-on_damp-25_move3_2024-10-02_04-42-29-349841/nn/last_RIGHT_1-freq_coll-on_damp-25_move3_ep_13000_rew_124.79679.pth"  # Update this path
        _, self.config_path = restore_file_from_wandb(
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-05_cup_fabric_reset-early_multigpu/files/runs/TOP_4-freq_coll-on_juno1_2_2024-10-07_23-27-58-967674/config_resolved.yaml?runName=TOP_4-freq_coll-on_juno1_2_2024-10-07_23-27-58-967674"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-05_cup_fabric_reset-early_multigpu/files/runs/LEFT_4-freq_juno2_2024-10-07_23-20-48-082226/config_resolved.yaml?runName=LEFT_4-freq_juno2_2024-10-07_23-20-48-082226"
            "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-06_cup_fabric_privileged_multigpu/files/runs/TOP_LSTM_DRSmall_ws-16_1gpu_2024-10-09_09-25-10-276712/config_resolved.yaml?runName=TOP_LSTM_DRSmall_ws-16_1gpu_2024-10-09_09-25-10-276712"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-06_cup_fabric_harder_privileged_multigpu/files/runs/LEFT_LSTM_Friction0-3_juno1_2024-10-09_09-03-57-666419/config_resolved.yaml?runName=LEFT_LSTM_Friction0-3_juno1_2024-10-09_09-03-57-666419"
        )
        _, self.checkpoint_path = restore_file_from_wandb(
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-05_cup_fabric_reset-early_multigpu/files/runs/TOP_4-freq_coll-on_juno1_2_2024-10-07_23-27-58-967674/nn/TOP_4-freq_coll-on_juno1_2.pth?runName=TOP_4-freq_coll-on_juno1_2_2024-10-07_23-27-58-967674"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-05_cup_fabric_reset-early_multigpu/files/runs/LEFT_4-freq_juno2_2024-10-07_23-20-48-082226/nn/LEFT_4-freq_juno2.pth?runName=LEFT_4-freq_juno2_2024-10-07_23-20-48-082226"
            "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-06_cup_fabric_privileged_multigpu/files/runs/TOP_LSTM_DRSmall_ws-16_1gpu_2024-10-09_09-25-10-276712/nn/TOP_LSTM_DRSmall_ws-16_1gpu.pth?runName=TOP_LSTM_DRSmall_ws-16_1gpu_2024-10-09_09-25-10-276712"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-06_cup_fabric_harder_privileged_multigpu/files/runs/LEFT_LSTM_Friction0-3_juno1_2024-10-09_09-03-57-666419/nn/LEFT_LSTM_Friction0-3_juno1.pth?runName=LEFT_LSTM_Friction0-3_juno1_2024-10-09_09-03-57-666419"
        )

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
            [0.0, -0.7, 0, -3.1416, -3.1416, -3.1416], device=self.device
        )
        self.palm_maxs = torch.tensor(
            [1.0, 0.7, 1.0, 3.1416, 3.1416, 3.1416], device=self.device
        )
        self.hand_mins = torch.tensor(
            [0.2475, -0.3286, -0.7238, -0.0192, -0.5532], device=self.device
        )
        self.hand_maxs = torch.tensor(
            [3.8336, 3.0025, 0.8977, 1.0243, 0.0629], device=self.device
        )

        self._setup_taskmap()
        self.t = 0

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
        self.received_fabric_state_time = rospy.Time.now()

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

        # Stop if the fabric states are not received for a long time
        assert self.received_fabric_state_time is not None
        MAX_DT_FABRIC_STATE_SEC = 1.0
        time_since_fabric_state = (
            rospy.Time.now() - self.received_fabric_state_time
        ).to_sec()
        if time_since_fabric_state > MAX_DT_FABRIC_STATE_SEC:
            log_msg = (
                f"Did not receive fabric states for {time_since_fabric_state} seconds"
            )
            rospy.logerr(log_msg)
            raise ValueError(log_msg)

        iiwa_joint_state_msg = copy.copy(self.iiwa_joint_state_msg)
        allegro_joint_state_msg = copy.copy(self.allegro_joint_state_msg)
        object_pose_msg = copy.copy(self.object_pose_msg)
        goal_object_pose_msg = copy.copy(self.goal_object_pose_msg)
        fabric_state_msg = copy.copy(self.fabric_state_msg)

        # Concatenate the data from joint states and object pose
        iiwa_position = np.array(iiwa_joint_state_msg.position)
        iiwa_velocity = np.array(iiwa_joint_state_msg.velocity)

        allegro_position = np.array(allegro_joint_state_msg.position)
        allegro_velocity = np.array(allegro_joint_state_msg.velocity)

        T_C_O = pose_msg_to_T(object_pose_msg)
        T_C_G = pose_msg_to_T(goal_object_pose_msg)

        if self.prev_object_pose_msg is not None:
            T_C_O_prev = pose_msg_to_T(self.prev_object_pose_msg)
        else:
            T_C_O_prev = T_C_O

        if self.prev_prev_object_pose_msg is not None:
            T_C_O_prev_prev = pose_msg_to_T(self.prev_prev_object_pose_msg)
        else:
            T_C_O_prev_prev = T_C_O_prev

        self.prev_prev_object_pose_msg = self.prev_object_pose_msg
        self.prev_object_pose_msg = object_pose_msg

        T_R_O = T_R_C @ T_C_O
        object_position_R, object_quat_xyzw_R = T_to_pos_quat_xyzw(T_R_O)

        T_R_G = T_R_C @ T_C_G
        goal_object_pos_R, goal_object_quat_xyzw_R = T_to_pos_quat_xyzw(T_R_G)

        T_R_O_prev = T_R_C @ T_C_O_prev
        object_position_R_prev, object_quat_xyzw_R_prev = T_to_pos_quat_xyzw(T_R_O_prev)

        T_R_O_prev_prev = T_R_C @ T_C_O_prev_prev
        object_position_R_prev_prev, object_quat_xyzw_R_prev_prev = T_to_pos_quat_xyzw(
            T_R_O_prev_prev
        )

        keypoint_offsets = to_torch(
            OBJECT_KEYPOINT_OFFSETS,
            device=self.device,
            dtype=torch.float,
        )
        assert_equals(keypoint_offsets.shape, (NUM_OBJECT_KEYPOINTS, NUM_XYZ))

        q = np.concatenate([iiwa_position, allegro_position])
        qd = np.concatenate([iiwa_velocity, allegro_velocity])

        fabric_q = np.array(fabric_state_msg.position)
        fabric_qd = np.array(fabric_state_msg.velocity)

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

        obs_dict["prev_object_pos"] = object_position_R_prev
        obs_dict["prev_object_quat_xyzw"] = object_quat_xyzw_R_prev
        obs_dict["prev_prev_object_pos"] = object_position_R_prev_prev
        obs_dict["prev_prev_object_quat_xyzw"] = object_quat_xyzw_R_prev_prev

        # object_keypoint_positions = (
        #     compute_keypoint_positions(
        #         pos=torch.tensor(object_position_R, device=self.device)
        #         .unsqueeze(0)
        #         .float(),
        #         quat_xyzw=torch.tensor(object_quat_xyzw_R, device=self.device)
        #         .unsqueeze(0)
        #         .float(),
        #         keypoint_offsets=keypoint_offsets.unsqueeze(0).float(),
        #     )
        #     .squeeze(0)
        #     .cpu()
        #     .numpy()
        # )
        # goal_object_keypoint_positions = (
        #     compute_keypoint_positions(
        #         pos=torch.tensor(goal_object_pos_R, device=self.device)
        #         .unsqueeze(0)
        #         .float(),
        #         quat_xyzw=torch.tensor(goal_object_quat_xyzw_R, device=self.device)
        #         .unsqueeze(0)
        #         .float(),
        #         keypoint_offsets=keypoint_offsets.unsqueeze(0).float(),
        #     )
        #     .squeeze(0)
        #     .cpu()
        #     .numpy()
        # )
        # object_vel = np.zeros(3)
        # object_angvel = np.zeros(3)
        # obs_dict["object_keypoint_positions"] = (
        #     object_keypoint_positions.reshape(
        #         NUM_OBJECT_KEYPOINTS * NUM_XYZ
        #     )
        # )
        # obs_dict["goal_object_keypoint_positions"] = (
        #     goal_object_keypoint_positions.reshape(
        #         NUM_OBJECT_KEYPOINTS * NUM_XYZ
        #     )
        # )
        # obs_dict["object_vel"] = object_vel
        # obs_dict["object_angvel"] = object_angvel
        # obs_dict["t"] = np.array([self.t]).reshape(1)
        # self.t += 1

        obs_dict["fabric_q"] = fabric_q
        obs_dict["fabric_qd"] = fabric_qd

        for k, v in obs_dict.items():
            assert len(v.shape) == 1, f"Shape of {k} is {v.shape}, expected 1D tensor"

        # DEBUG
        # for k, v in obs_dict.items():
        #     print(f"{k}: {v}")
        # print()

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
        N = action.shape[0]
        assert_equals(action.shape, (N, self.num_actions))

        # Rescale the normalized actions from [-1, 1] to the actual target ranges
        palm_target = rescale(
            values=action[:, :6],
            old_mins=torch.ones_like(self.palm_mins) * -1,
            old_maxs=torch.ones_like(self.palm_maxs) * 1,
            new_mins=self.palm_mins,
            new_maxs=self.palm_maxs,
        )
        hand_target = rescale(
            values=action[:, 6:],
            old_mins=torch.ones_like(self.hand_mins) * -1,
            old_maxs=torch.ones_like(self.hand_maxs) * 1,
            new_mins=self.hand_mins,
            new_maxs=self.hand_maxs,
        )
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
                assert_equals(obs.shape, (1, self.num_observations))

                # Get the normalized action from the RL player
                normalized_action = self.player.get_normalized_action(obs=obs)
                # normalized_action = torch.zeros(1, self.num_actions, device=self.device)
                assert_equals(normalized_action.shape, (1, self.num_actions))

                # Rescale the action to get palm and hand targets
                palm_target, hand_target = self.rescale_action(normalized_action)
                assert_equals(palm_target.shape, (1, 6))
                assert_equals(hand_target.shape, (1, 5))
                palm_target = palm_target.squeeze(0)
                hand_target = hand_target.squeeze(0)

                # DEBUG
                # print(f"normalized_action: {normalized_action}")
                # print(f"palm_target: {palm_target}")
                # print(f"hand_target: {hand_target}")
                # print()

                # Publish the targets
                self.publish_targets(palm_target, hand_target)

            # Sleep to maintain 15 loop rate
            before_sleep_time = rospy.Time.now()
            self.rate.sleep()
            after_sleep_time = rospy.Time.now()
            rospy.loginfo(
                f"{rospy.get_name()} Max rate: {1 / (before_sleep_time - start_time).to_sec()} Hz ({(before_sleep_time - start_time).to_sec() * 1000}ms), Actual rate: {1 / (after_sleep_time - start_time).to_sec()} Hz"
            )


if __name__ == "__main__":
    try:
        rl_policy_node = RLPolicyNode()
        rl_policy_node.run()
    except rospy.ROSInterruptException:
        pass
