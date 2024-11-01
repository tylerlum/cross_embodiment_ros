#!/usr/bin/env python

import copy
from pathlib import Path
from typing import Optional, Tuple, Literal
import functools

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Pose
from isaacgym.torch_utils import to_torch
from isaacgymenvs.utils.cross_embodiment.camera_extrinsics import (
    ZED_CAMERA_T_R_C,
    REALSENSE_CAMERA_T_R_C,
)
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
from isaacgymenvs.utils.wandb_utils import restore_file_from_wandb
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout

from print_utils import get_ros_loop_rate_str
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


FABRIC_MODE: Literal["PCA", "ALL"] = "PCA"


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

        # RL Player setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_observations = 144  # Update this number based on actual dimensions
        if FABRIC_MODE == "PCA":
            self.num_actions = 11  # First 6 for palm, last 5 for hand
        elif FABRIC_MODE == "ALL":
            self.num_actions = 22  # First 6 for palm, last 16 for hand
        else:
            raise ValueError(f"Invalid FABRIC_MODE: {FABRIC_MODE}")

        # self.config_path = "/move/u/tylerlum/github_repos/bidexhands_isaacgymenvs/isaacgymenvs/runs/RIGHT_1-freq_coll-on_damp-25_move3_2024-10-02_04-42-29-349841/config_resolved.yaml"  # Update this path
        # self.checkpoint_path = "/move/u/tylerlum/github_repos/bidexhands_isaacgymenvs/isaacgymenvs/runs/RIGHT_1-freq_coll-on_damp-25_move3_2024-10-02_04-42-29-349841/nn/last_RIGHT_1-freq_coll-on_damp-25_move3_ep_13000_rew_124.79679.pth"  # Update this path
        # _, self.config_path = restore_file_from_wandb(
        #     # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-05_cup_fabric_reset-early_multigpu/files/runs/TOP_4-freq_coll-on_juno1_2_2024-10-07_23-27-58-967674/config_resolved.yaml?runName=TOP_4-freq_coll-on_juno1_2_2024-10-07_23-27-58-967674"
        #     # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-05_cup_fabric_reset-early_multigpu/files/runs/LEFT_4-freq_juno2_2024-10-07_23-20-48-082226/config_resolved.yaml?runName=LEFT_4-freq_juno2_2024-10-07_23-20-48-082226"
        #     "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-06_cup_fabric_privileged_multigpu/files/runs/TOP_LSTM_DRSmall_ws-16_1gpu_2024-10-09_09-25-10-276712/config_resolved.yaml?runName=TOP_LSTM_DRSmall_ws-16_1gpu_2024-10-09_09-25-10-276712"
        #     # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-06_cup_fabric_harder_privileged_multigpu/files/runs/LEFT_LSTM_Friction0-3_juno1_2024-10-09_09-03-57-666419/config_resolved.yaml?runName=LEFT_LSTM_Friction0-3_juno1_2024-10-09_09-03-57-666419"
        # )
        # _, self.checkpoint_path = restore_file_from_wandb(
        #     # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-05_cup_fabric_reset-early_multigpu/files/runs/TOP_4-freq_coll-on_juno1_2_2024-10-07_23-27-58-967674/nn/TOP_4-freq_coll-on_juno1_2.pth?runName=TOP_4-freq_coll-on_juno1_2_2024-10-07_23-27-58-967674"
        #     # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-05_cup_fabric_reset-early_multigpu/files/runs/LEFT_4-freq_juno2_2024-10-07_23-20-48-082226/nn/LEFT_4-freq_juno2.pth?runName=LEFT_4-freq_juno2_2024-10-07_23-20-48-082226"
        #     "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-06_cup_fabric_privileged_multigpu/files/runs/TOP_LSTM_DRSmall_ws-16_1gpu_2024-10-09_09-25-10-276712/nn/TOP_LSTM_DRSmall_ws-16_1gpu.pth?runName=TOP_LSTM_DRSmall_ws-16_1gpu_2024-10-09_09-25-10-276712"
        #     # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-06_cup_fabric_harder_privileged_multigpu/files/runs/LEFT_LSTM_Friction0-3_juno1_2024-10-09_09-03-57-666419/nn/LEFT_LSTM_Friction0-3_juno1.pth?runName=LEFT_LSTM_Friction0-3_juno1_2024-10-09_09-03-57-666419"
        # )
        # self.config_path = "/juno/u/tylerlum/Downloads/RIGHT_coll-off_juno2_1gpu_2024-10-15_15-50-39-601698/config_resolved.yaml"
        # self.checkpoint_path = "/juno/u/tylerlum/Downloads/RIGHT_coll-off_juno2_1gpu_2024-10-15_15-50-39-601698/RIGHT_coll-off_juno2_1gpu.pth"
        # self.config_path = "/juno/u/tylerlum/Downloads/LEFT_coll-off_juno2_1gpu_2024-10-15_15-52-42-495096/config_resolved.yaml"
        # self.checkpoint_path = "/juno/u/tylerlum/Downloads/LEFT_coll-off_juno2_1gpu_2024-10-15_15-52-42-495096/LEFT_coll-off_juno2_1gpu.pth"
        _, self.config_path = restore_file_from_wandb(
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-16_crackerbox_fabric_hard/files/runs/lstmnoconcat_move1_2024-10-16_02-00-31-760423/config_resolved.yaml?runName=lstmnoconcat_move1_2024-10-16_02-00-31-760423"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-16_crackerbox_fabric_hard/files/runs/noise0-02_move2_2024-10-16_02-04-04-538040/config_resolved.yaml?runName=noise0-02_move2_2024-10-16_02-04-04-538040"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-16_crackerbox_fabric_hard/files/runs/friction0-3_juno2_2024-10-16_01-56-46-200454/config_resolved.yaml?runName=friction0-3_juno2_2024-10-16_01-56-46-200454"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-16_crackerbox_fabric_hard/files/runs/base_run_juno2_2024-10-16_01-49-18-084062/config_resolved.yaml?runName=base_run_juno2_2024-10-16_01-49-18-084062"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-16_crackerbox_fabric_hard/files/runs/succ0-05_move2_2024-10-16_02-04-04-537736/config_resolved.yaml?runName=succ0-05_move2_2024-10-16_02-04-04-537736"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-16_crackerbox_fabric_hard/files/runs/DRLarge_friction0-5_juno2_2024-10-16_01-56-25-580392/config_resolved.yaml?runName=DRLarge_friction0-5_juno2_2024-10-16_01-56-25-580392"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-17_crackerbox_fabric_friction_mass_inertia/files/runs/heavy_run_juno2_2024-10-16_23-33-13-473749/config_resolved.yaml?runName=heavy_run_juno2_2024-10-16_23-33-13-473749"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-17_crackerbox_fabric_friction_mass_inertia/files/runs/base_run_juno2_2024-10-16_23-33-13-473827/config_resolved.yaml?runName=base_run_juno2_2024-10-16_23-33-13-473827"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-17_crackerbox_fabric_friction_mass_inertia/files/runs/friction0-3_juno2_2024-10-16_23-33-13-473890/config_resolved.yaml?runName=friction0-3_juno2_2024-10-16_23-33-13-473890"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-17_crackerbox_fabric_friction_mass_inertia/files/runs/heavy_run_friction0-5_move1_2024-10-16_23-42-30-218923/config_resolved.yaml?runName=heavy_run_friction0-5_move1_2024-10-16_23-42-30-218923"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-17_crackerbox_fabric_friction_mass_inertia/files/runs/orient0_juno1_2024-10-16_23-45-43-871057/config_resolved.yaml?runName=orient0_juno1_2024-10-16_23-45-43-871057"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-21_crackerbox/files/runs/damp45_ws-19_2024-10-21_02-22-24-946771/config_resolved.yaml?runName=damp45_ws-19_2024-10-21_02-22-24-946771"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-18_crackerbox/files/runs/friction0-5_move1_2024-10-18_01-20-15-137476/config_resolved.yaml?runName=friction0-5_move1_2024-10-18_01-20-15-137476"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-21_crackerbox/files/runs/freq1_move2_2024-10-21_02-18-40-861441/config_resolved.yaml?runName=freq1_move2_2024-10-21_02-18-40-861441"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-18_crackerbox/files/runs/noise0-05_move1_2024-10-18_01-20-15-137825/config_resolved.yaml?runName=noise0-05_move1_2024-10-18_01-20-15-137825"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-21_crackerbox/files/runs/base_move2_2024-10-21_02-18-40-861433/config_resolved.yaml?runName=base_move2_2024-10-21_02-18-40-861433"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-21_crackerbox/files/runs/coll-off_move1_2024-10-21_02-18-42-117793/config_resolved.yaml?runName=coll-off_move1_2024-10-21_02-18-42-117793"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-22_crackerbox_trajectory/files/runs/lift_damp45_move1_2024-10-22_00-20-06-579299/config_resolved.yaml?runName=lift_damp45_move1_2024-10-22_00-20-06-579299"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-22_crackerbox_trajectory/files/runs/lift_move2_2024-10-22_00-20-08-861728/config_resolved.yaml?runName=lift_move2_2024-10-22_00-20-08-861728"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-22_crackerbox_trajectory/files/runs/push_damp45_ws-16_2024-10-22_00-30-45-714566/config_resolved.yaml?runName=push_damp45_ws-16_2024-10-22_00-30-45-714566"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-22_crackerbox_trajectory/files/runs/diagonal_push_damp45_move1_2024-10-22_00-20-06-579262/config_resolved.yaml?runName=diagonal_push_damp45_move1_2024-10-22_00-20-06-579262"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-23_crackerbox_trajectory_harder/files/runs/lift_base_move1_2024-10-23_03-23-20-279676/config_resolved.yaml?runName=lift_base_move1_2024-10-23_03-23-20-279676"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-23_crackerbox_trajectory_harder/files/runs/lift_friction0-5_freq1_move2_2024-10-23_03-35-15-483418/config_resolved.yaml?runName=lift_friction0-5_freq1_move2_2024-10-23_03-35-15-483418"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-23_crackerbox_trajectory_harder/files/runs/push_friction0-3_move3_2024-10-23_03-44-30-072131/config_resolved.yaml?runName=push_friction0-3_move3_2024-10-23_03-44-30-072131"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-23_crackerbox_trajectory_harder/files/runs/lift_friction0-5_DRLarge_force0-1_noise0-02_move1_2024-10-23_03-32-30-289040/config_resolved.yaml?runName=lift_friction0-5_DRLarge_force0-1_noise0-02_move1_2024-10-23_03-32-30-289040"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-23_crackerbox_trajectory_harder/files/runs/lift_friction0-5_move2_2024-10-23_03-35-31-266374/config_resolved.yaml?runName=lift_friction0-5_move2_2024-10-23_03-35-31-266374"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-23_crackerbox_trajectory_harder/files/runs/push_friction0-5_freq1_juno2_2024-10-23_03-41-46-838160/config_resolved.yaml?runName=push_friction0-5_freq1_juno2_2024-10-23_03-41-46-838160"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-23_crackerbox_trajectory_harder/files/runs/push_friction0-5_DRLarge_force0-1_noise0-02_ws-16_2024-10-23_03-40-11-262111/config_resolved.yaml?runName=push_friction0-5_DRLarge_force0-1_noise0-02_ws-16_2024-10-23_03-40-11-262111"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-23_crackerbox_trajectory_harder/files/runs/lift_friction0-5_allhand_move1_2024-10-23_03-32-30-260271/config_resolved.yaml?runName=lift_friction0-5_allhand_move1_2024-10-23_03-32-30-260271"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-23_crackerbox_trajectory_harder/files/runs/push_friction0-5_allhand_ws-19_2024-10-23_03-38-10-555052/config_resolved.yaml?runName=push_friction0-5_allhand_ws-19_2024-10-23_03-38-10-555052"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/9_base_ws-19_2024-10-25_02-03-28-531680/config_resolved.yaml?runName=9_base_ws-19_2024-10-25_02-03-28-531680"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/9_friction0-5_force0-1_noise0-02_juno1_2024-10-25_02-02-47-479939/config_resolved.yaml?runName=9_friction0-5_force0-1_noise0-02_juno1_2024-10-25_02-02-47-479939"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/10_base_ws-16_2024-10-25_02-06-37-140026/config_resolved.yaml?runName=10_base_ws-16_2024-10-25_02-06-37-140026"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/11_friction0-5_move2_2024-10-25_02-00-00-650467/config_resolved.yaml?runName=11_friction0-5_move2_2024-10-25_02-00-00-650467"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/11_friction0-5_force0-1_noise0-02_move2_2024-10-25_02-00-00-650319/config_resolved.yaml?runName=11_friction0-5_force0-1_noise0-02_move2_2024-10-25_02-00-00-650319"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/8_friction0-5_juno2_2024-10-25_02-00-11-785544/config_resolved.yaml?runName=8_friction0-5_juno2_2024-10-25_02-00-11-785544"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/8_friction0-5_force0-1_noise0-02_move1_2024-10-25_01-59-45-381063/config_resolved.yaml?runName=8_friction0-5_force0-1_noise0-02_move1_2024-10-25_01-59-45-381063"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/8_base_move1_2024-10-25_01-59-45-381051/config_resolved.yaml?runName=8_base_move1_2024-10-25_01-59-45-381051"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/8_friction0-3_move1_2024-10-25_01-59-45-381040/config_resolved.yaml?runName=8_friction0-3_move1_2024-10-25_01-59-45-381040"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/11_base_juno2_2024-10-25_02-00-20-181080/config_resolved.yaml?runName=11_base_juno2_2024-10-25_02-00-20-181080"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/9_friction0-5_force0-1_noise0-02_juno1_2024-10-25_02-02-47-479939/config_resolved.yaml?runName=9_friction0-5_force0-1_noise0-02_juno1_2024-10-25_02-02-47-479939"
            # 2024-10-29
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/11_base_juno2_2024-10-25_02-00-20-181080/config_resolved.yaml?runName=11_base_juno2_2024-10-25_02-00-20-181080"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/11_friction0-5_move2_2024-10-25_02-00-00-650467/config_resolved.yaml?runName=11_friction0-5_move2_2024-10-25_02-00-00-650467"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/11_friction0-5_force0-1_noise0-02_move2_2024-10-25_02-00-00-650319/config_resolved.yaml?runName=11_friction0-5_force0-1_noise0-02_move2_2024-10-25_02-00-00-650319"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/9_base_ws-19_2024-10-25_02-03-28-531680/config_resolved.yaml?runName=9_base_ws-19_2024-10-25_02-03-28-531680"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/9_friction0-5_juno1_2024-10-25_02-02-31-460085/config_resolved.yaml?runName=9_friction0-5_juno1_2024-10-25_02-02-31-460085"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/9_friction0-5_force0-1_noise0-02_juno1_2024-10-25_02-02-47-479939/config_resolved.yaml?runName=9_friction0-5_force0-1_noise0-02_juno1_2024-10-25_02-02-47-479939"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/8_base_move1_2024-10-25_01-59-45-381051/config_resolved.yaml?runName=8_base_move1_2024-10-25_01-59-45-381051"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/8_friction0-5_juno2_2024-10-25_02-00-11-785544/config_resolved.yaml?runName=8_friction0-5_juno2_2024-10-25_02-00-11-785544"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/8_friction0-5_force0-1_noise0-02_move1_2024-10-25_01-59-45-381063/config_resolved.yaml?runName=8_friction0-5_force0-1_noise0-02_move1_2024-10-25_01-59-45-381063"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-28_crackerbox_hard_trajectory/files/runs/7_base_move2_2024-10-28_01-06-50-382887/config_resolved.yaml?runName=7_base_move2_2024-10-28_01-06-50-382887"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-28_crackerbox_hard_trajectory/files/runs/7_friction0-5_move2_2024-10-28_01-06-50-383023/config_resolved.yaml?runName=7_friction0-5_move2_2024-10-28_01-06-50-383023"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-28_crackerbox_hard_trajectory/files/runs/6_base_move1_2024-10-28_01-06-14-675800/config_resolved.yaml?runName=6_base_move1_2024-10-28_01-06-14-675800"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-28_crackerbox_hard_trajectory/files/runs/6_friction0-5_force0-1_noise0-02_juno2_2024-10-28_01-10-01-351943/config_resolved.yaml?runName=6_friction0-5_force0-1_noise0-02_juno2_2024-10-28_01-10-01-351943"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-28_crackerbox_hard_trajectory/files/runs/4_base_move1_2024-10-28_01-05-18-652438/config_resolved.yaml?runName=4_base_move1_2024-10-28_01-05-18-652438"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-28_crackerbox_hard_trajectory/files/runs/4_friction0-5_move1_2024-10-28_01-05-18-652390/config_resolved.yaml?runName=4_friction0-5_move1_2024-10-28_01-05-18-652390"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-28_crackerbox_hard_trajectory/files/runs/4_friction0-5_force0-1_noise0-02_juno2_2024-10-28_01-10-01-351869/config_resolved.yaml?runName=4_friction0-5_force0-1_noise0-02_juno2_2024-10-28_01-10-01-351869"
            # 2024-10-31
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-31_cup_trajectory/files/runs/0_base_move1_2024-10-31_01-47-27-815811/config_resolved.yaml?runName=0_base_move1_2024-10-31_01-47-27-815811"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-31_cup_trajectory/files/runs/8_base_juno2_2024-10-31_01-49-07-406102/config_resolved.yaml?runName=8_base_juno2_2024-10-31_01-49-07-406102"
            "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-31_cup_trajectory/files/runs/8_friction0-5_ws-19_2024-10-31_01-58-35-276051/config_resolved.yaml?runName=8_friction0-5_ws-19_2024-10-31_01-58-35-276051"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-31_cup_trajectory/files/runs/7_base_juno2_2024-10-31_01-49-04-386494/config_resolved.yaml?runName=7_base_juno2_2024-10-31_01-49-04-386494"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-31_cup_trajectory/files/runs/8_friction0-3-5-8_ws-16_2024-10-31_02-00-48-336319/config_resolved.yaml?runName=8_friction0-3-5-8_ws-16_2024-10-31_02-00-48-336319"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-31_cup_trajectory/files/runs/5_base_juno2_2024-10-31_01-49-04-386691/config_resolved.yaml?runName=5_base_juno2_2024-10-31_01-49-04-386691"
        )
        _, self.checkpoint_path = restore_file_from_wandb(
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-16_crackerbox_fabric_hard/files/runs/lstmnoconcat_move1_2024-10-16_02-00-31-760423/nn/lstmnoconcat_move1.pth?runName=lstmnoconcat_move1_2024-10-16_02-00-31-760423"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-16_crackerbox_fabric_hard/files/runs/noise0-02_move2_2024-10-16_02-04-04-538040/nn/noise0-02_move2.pth?runName=noise0-02_move2_2024-10-16_02-04-04-538040"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-16_crackerbox_fabric_hard/files/runs/friction0-3_juno2_2024-10-16_01-56-46-200454/nn/friction0-3_juno2.pth?runName=friction0-3_juno2_2024-10-16_01-56-46-200454"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-16_crackerbox_fabric_hard/files/runs/base_run_juno2_2024-10-16_01-49-18-084062/nn/base_run_juno2.pth?runName=base_run_juno2_2024-10-16_01-49-18-084062"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-16_crackerbox_fabric_hard/files/runs/succ0-05_move2_2024-10-16_02-04-04-537736/nn/succ0-05_move2.pth?runName=succ0-05_move2_2024-10-16_02-04-04-537736"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-16_crackerbox_fabric_hard/files/runs/DRLarge_friction0-5_juno2_2024-10-16_01-56-25-580392/nn/DRLarge_friction0-5_juno2.pth?runName=DRLarge_friction0-5_juno2_2024-10-16_01-56-25-580392"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-17_crackerbox_fabric_friction_mass_inertia/files/runs/heavy_run_juno2_2024-10-16_23-33-13-473749/nn/heavy_run_juno2.pth?runName=heavy_run_juno2_2024-10-16_23-33-13-473749"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-17_crackerbox_fabric_friction_mass_inertia/files/runs/base_run_juno2_2024-10-16_23-33-13-473827/nn/base_run_juno2.pth?runName=base_run_juno2_2024-10-16_23-33-13-473827"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-17_crackerbox_fabric_friction_mass_inertia/files/runs/friction0-3_juno2_2024-10-16_23-33-13-473890/nn/friction0-3_juno2.pth?runName=friction0-3_juno2_2024-10-16_23-33-13-473890"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-17_crackerbox_fabric_friction_mass_inertia/files/runs/heavy_run_friction0-5_move1_2024-10-16_23-42-30-218923/nn/heavy_run_friction0-5_move1.pth?runName=heavy_run_friction0-5_move1_2024-10-16_23-42-30-218923"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-17_crackerbox_fabric_friction_mass_inertia/files/runs/orient0_juno1_2024-10-16_23-45-43-871057/nn/orient0_juno1.pth?runName=orient0_juno1_2024-10-16_23-45-43-871057"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-21_crackerbox/files/runs/damp45_ws-19_2024-10-21_02-22-24-946771/nn/damp45_ws-19.pth?runName=damp45_ws-19_2024-10-21_02-22-24-946771"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-18_crackerbox/files/runs/friction0-5_move1_2024-10-18_01-20-15-137476/nn/friction0-5_move1.pth?runName=friction0-5_move1_2024-10-18_01-20-15-137476"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-21_crackerbox/files/runs/freq1_move2_2024-10-21_02-18-40-861441/nn/freq1_move2.pth?runName=freq1_move2_2024-10-21_02-18-40-861441"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-18_crackerbox/files/runs/noise0-05_move1_2024-10-18_01-20-15-137825/nn/noise0-05_move1.pth?runName=noise0-05_move1_2024-10-18_01-20-15-137825"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-21_crackerbox/files/runs/base_move2_2024-10-21_02-18-40-861433/nn/base_move2.pth?runName=base_move2_2024-10-21_02-18-40-861433"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-21_crackerbox/files/runs/coll-off_move1_2024-10-21_02-18-42-117793/nn/coll-off_move1.pth?runName=coll-off_move1_2024-10-21_02-18-42-117793"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-22_crackerbox_trajectory/files/runs/lift_damp45_move1_2024-10-22_00-20-06-579299/nn/lift_damp45_move1.pth?runName=lift_damp45_move1_2024-10-22_00-20-06-579299"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-22_crackerbox_trajectory/files/runs/lift_move2_2024-10-22_00-20-08-861728/nn/lift_move2.pth?runName=lift_move2_2024-10-22_00-20-08-861728"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-22_crackerbox_trajectory/files/runs/push_damp45_ws-16_2024-10-22_00-30-45-714566/nn/push_damp45_ws-16.pth?runName=push_damp45_ws-16_2024-10-22_00-30-45-714566"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-22_crackerbox_trajectory/files/runs/diagonal_push_damp45_move1_2024-10-22_00-20-06-579262/nn/diagonal_push_damp45_move1.pth?runName=diagonal_push_damp45_move1_2024-10-22_00-20-06-579262"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-23_crackerbox_trajectory_harder/files/runs/lift_base_move1_2024-10-23_03-23-20-279676/nn/lift_base_move1.pth?runName=lift_base_move1_2024-10-23_03-23-20-279676"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-23_crackerbox_trajectory_harder/files/runs/lift_friction0-5_freq1_move2_2024-10-23_03-35-15-483418/nn/lift_friction0-5_freq1_move2.pth?runName=lift_friction0-5_freq1_move2_2024-10-23_03-35-15-483418"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-23_crackerbox_trajectory_harder/files/runs/push_friction0-3_move3_2024-10-23_03-44-30-072131/nn/push_friction0-3_move3.pth?runName=push_friction0-3_move3_2024-10-23_03-44-30-072131"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-23_crackerbox_trajectory_harder/files/runs/lift_friction0-5_DRLarge_force0-1_noise0-02_move1_2024-10-23_03-32-30-289040/nn/lift_friction0-5_DRLarge_force0-1_noise0-02_move1.pth?runName=lift_friction0-5_DRLarge_force0-1_noise0-02_move1_2024-10-23_03-32-30-289040"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-23_crackerbox_trajectory_harder/files/runs/lift_friction0-5_move2_2024-10-23_03-35-31-266374/nn/lift_friction0-5_move2.pth?runName=lift_friction0-5_move2_2024-10-23_03-35-31-266374"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-23_crackerbox_trajectory_harder/files/runs/push_friction0-5_freq1_juno2_2024-10-23_03-41-46-838160/nn/push_friction0-5_freq1_juno2.pth?runName=push_friction0-5_freq1_juno2_2024-10-23_03-41-46-838160"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-23_crackerbox_trajectory_harder/files/runs/push_friction0-5_DRLarge_force0-1_noise0-02_ws-16_2024-10-23_03-40-11-262111/nn/push_friction0-5_DRLarge_force0-1_noise0-02_ws-16.pth?runName=push_friction0-5_DRLarge_force0-1_noise0-02_ws-16_2024-10-23_03-40-11-262111"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-23_crackerbox_trajectory_harder/files/runs/lift_friction0-5_allhand_move1_2024-10-23_03-32-30-260271/nn/lift_friction0-5_allhand_move1.pth?runName=lift_friction0-5_allhand_move1_2024-10-23_03-32-30-260271"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-23_crackerbox_trajectory_harder/files/runs/push_friction0-5_allhand_ws-19_2024-10-23_03-38-10-555052/nn/push_friction0-5_allhand_ws-19.pth?runName=push_friction0-5_allhand_ws-19_2024-10-23_03-38-10-555052"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/9_base_ws-19_2024-10-25_02-03-28-531680/nn/9_base_ws-19.pth?runName=9_base_ws-19_2024-10-25_02-03-28-531680"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/9_friction0-5_force0-1_noise0-02_juno1_2024-10-25_02-02-47-479939/nn/9_friction0-5_force0-1_noise0-02_juno1.pth?runName=9_friction0-5_force0-1_noise0-02_juno1_2024-10-25_02-02-47-479939"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/10_base_ws-16_2024-10-25_02-06-37-140026/nn/10_base_ws-16.pth?runName=10_base_ws-16_2024-10-25_02-06-37-140026"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/11_friction0-5_move2_2024-10-25_02-00-00-650467/nn/11_friction0-5_move2.pth?runName=11_friction0-5_move2_2024-10-25_02-00-00-650467"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/11_friction0-5_force0-1_noise0-02_move2_2024-10-25_02-00-00-650319/nn/11_friction0-5_force0-1_noise0-02_move2.pth?runName=11_friction0-5_force0-1_noise0-02_move2_2024-10-25_02-00-00-650319"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/8_friction0-5_juno2_2024-10-25_02-00-11-785544/nn/8_friction0-5_juno2.pth?runName=8_friction0-5_juno2_2024-10-25_02-00-11-785544"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/8_friction0-5_force0-1_noise0-02_move1_2024-10-25_01-59-45-381063/nn/8_friction0-5_force0-1_noise0-02_move1.pth?runName=8_friction0-5_force0-1_noise0-02_move1_2024-10-25_01-59-45-381063"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/8_base_move1_2024-10-25_01-59-45-381051/nn/8_base_move1.pth?runName=8_base_move1_2024-10-25_01-59-45-381051"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/8_friction0-3_move1_2024-10-25_01-59-45-381040/nn/8_friction0-3_move1.pth?runName=8_friction0-3_move1_2024-10-25_01-59-45-381040"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/11_base_juno2_2024-10-25_02-00-20-181080/nn/11_base_juno2.pth?runName=11_base_juno2_2024-10-25_02-00-20-181080"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/9_friction0-5_force0-1_noise0-02_juno1_2024-10-25_02-02-47-479939/nn/9_friction0-5_force0-1_noise0-02_juno1.pth?runName=9_friction0-5_force0-1_noise0-02_juno1_2024-10-25_02-02-47-479939"
            # 2024-10-29
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/11_base_juno2_2024-10-25_02-00-20-181080/nn/11_base_juno2.pth?runName=11_base_juno2_2024-10-25_02-00-20-181080"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/11_friction0-5_move2_2024-10-25_02-00-00-650467/nn/11_friction0-5_move2.pth?runName=11_friction0-5_move2_2024-10-25_02-00-00-650467"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/11_friction0-5_force0-1_noise0-02_move2_2024-10-25_02-00-00-650319/nn/11_friction0-5_force0-1_noise0-02_move2.pth?runName=11_friction0-5_force0-1_noise0-02_move2_2024-10-25_02-00-00-650319"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/9_base_ws-19_2024-10-25_02-03-28-531680/nn/9_base_ws-19.pth?runName=9_base_ws-19_2024-10-25_02-03-28-531680"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/9_friction0-5_juno1_2024-10-25_02-02-31-460085/nn/9_friction0-5_juno1.pth?runName=9_friction0-5_juno1_2024-10-25_02-02-31-460085"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/9_friction0-5_force0-1_noise0-02_juno1_2024-10-25_02-02-47-479939/nn/9_friction0-5_force0-1_noise0-02_juno1.pth?runName=9_friction0-5_force0-1_noise0-02_juno1_2024-10-25_02-02-47-479939"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/8_base_move1_2024-10-25_01-59-45-381051/nn/8_base_move1.pth?runName=8_base_move1_2024-10-25_01-59-45-381051"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/8_friction0-5_juno2_2024-10-25_02-00-11-785544/nn/8_friction0-5_juno2.pth?runName=8_friction0-5_juno2_2024-10-25_02-00-11-785544"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-25_crackerbox_new_trajectory/files/runs/8_friction0-5_force0-1_noise0-02_move1_2024-10-25_01-59-45-381063/nn/8_friction0-5_force0-1_noise0-02_move1.pth?runName=8_friction0-5_force0-1_noise0-02_move1_2024-10-25_01-59-45-381063"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-28_crackerbox_hard_trajectory/files/runs/7_base_move2_2024-10-28_01-06-50-382887/nn/7_base_move2.pth?runName=7_base_move2_2024-10-28_01-06-50-382887"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-28_crackerbox_hard_trajectory/files/runs/7_friction0-5_move2_2024-10-28_01-06-50-383023/nn/7_friction0-5_move2.pth?runName=7_friction0-5_move2_2024-10-28_01-06-50-383023"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-28_crackerbox_hard_trajectory/files/runs/6_base_move1_2024-10-28_01-06-14-675800/nn/6_base_move1.pth?runName=6_base_move1_2024-10-28_01-06-14-675800"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-28_crackerbox_hard_trajectory/files/runs/6_friction0-5_force0-1_noise0-02_juno2_2024-10-28_01-10-01-351943/nn/6_friction0-5_force0-1_noise0-02_juno2.pth?runName=6_friction0-5_force0-1_noise0-02_juno2_2024-10-28_01-10-01-351943"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-28_crackerbox_hard_trajectory/files/runs/4_base_move1_2024-10-28_01-05-18-652438/nn/4_base_move1.pth?runName=4_base_move1_2024-10-28_01-05-18-652438"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-28_crackerbox_hard_trajectory/files/runs/4_friction0-5_move1_2024-10-28_01-05-18-652390/nn/4_friction0-5_move1.pth?runName=4_friction0-5_move1_2024-10-28_01-05-18-652390"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-28_crackerbox_hard_trajectory/files/runs/4_friction0-5_force0-1_noise0-02_juno2_2024-10-28_01-10-01-351869/nn/4_friction0-5_force0-1_noise0-02_juno2.pth?runName=4_friction0-5_force0-1_noise0-02_juno2_2024-10-28_01-10-01-351869"
            # 2024-10-31
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-31_cup_trajectory/files/runs/0_base_move1_2024-10-31_01-47-27-815811/nn/0_base_move1.pth?runName=0_base_move1_2024-10-31_01-47-27-815811"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-31_cup_trajectory/files/runs/8_base_juno2_2024-10-31_01-49-07-406102/nn/8_base_juno2.pth?runName=8_base_juno2_2024-10-31_01-49-07-406102"
            "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-31_cup_trajectory/files/runs/8_friction0-5_ws-19_2024-10-31_01-58-35-276051/nn/8_friction0-5_ws-19.pth?runName=8_friction0-5_ws-19_2024-10-31_01-58-35-276051"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-31_cup_trajectory/files/runs/7_base_juno2_2024-10-31_01-49-04-386494/nn/7_base_juno2.pth?runName=7_base_juno2_2024-10-31_01-49-04-386494"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-31_cup_trajectory/files/runs/8_friction0-3-5-8_ws-16_2024-10-31_02-00-48-336319/nn/8_friction0-3-5-8_ws-16.pth?runName=8_friction0-3-5-8_ws-16_2024-10-31_02-00-48-336319"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-31_cup_trajectory/files/runs/5_base_juno2_2024-10-31_01-49-04-386691/nn/5_base_juno2.pth?runName=5_base_juno2_2024-10-31_01-49-04-386691"
        )

        # Create the RL player
        self.player = RlPlayer(
            num_observations=self.num_observations,
            num_actions=self.num_actions,
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

        # ROS rate
        # self.rate_hz = 15
        # self.rate_hz = 60
        control_dt = (
            self.player.cfg["task"]["env"]["controlFrequencyInv"]
            * self.player.cfg["task"]["sim"]["dt"]
        )
        self.rate_hz = 1.0 / control_dt
        self.rate = rospy.Rate(self.rate_hz)

        # Define limits for palm and hand targets
        self.palm_mins = torch.tensor(
            [0.0, -0.7, 0, -3.1416, -3.1416, -3.1416], device=self.device
        )
        self.palm_maxs = torch.tensor(
            [1.0, 0.7, 1.0, 3.1416, 3.1416, 3.1416], device=self.device
        )

        hand_action_space = self.player.cfg["task"]["env"]["custom"][
            "FABRIC_HAND_ACTION_SPACE"
        ]
        assert (
            hand_action_space == FABRIC_MODE
        ), f"Invalid hand action space: {hand_action_space} != {FABRIC_MODE}"
        if FABRIC_MODE == "PCA":
            self.hand_mins = torch.tensor(
                [0.2475, -0.3286, -0.7238, -0.0192, -0.5532], device=self.device
            )
            self.hand_maxs = torch.tensor(
                [3.8336, 3.0025, 0.8977, 1.0243, 0.0629], device=self.device
            )
        elif FABRIC_MODE == "ALL":
            self.hand_mins = torch.tensor(
                [
                    -0.4700,
                    -0.1960,
                    -0.1740,
                    -0.2270,
                    -0.4700,
                    -0.1960,
                    -0.1740,
                    -0.2270,
                    -0.4700,
                    -0.1960,
                    -0.1740,
                    -0.2270,
                    0.2630,
                    -0.1050,
                    -0.1890,
                    -0.1620,
                ],
                device=self.device,
            )
            self.hand_maxs = torch.tensor(
                [
                    0.4700,
                    1.6100,
                    1.7090,
                    1.6180,
                    0.4700,
                    1.6100,
                    1.7090,
                    1.6180,
                    0.4700,
                    1.6100,
                    1.7090,
                    1.6180,
                    1.3960,
                    1.1630,
                    1.6440,
                    1.7190,
                ],
                device=self.device,
            )
        else:
            raise ValueError(f"Invalid FABRIC_MODE = {FABRIC_MODE}")

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

        T_R_O = self.T_R_C @ T_C_O
        object_position_R, object_quat_xyzw_R = T_to_pos_quat_xyzw(T_R_O)

        T_R_G = self.T_R_C @ T_C_G
        goal_object_pos_R, goal_object_quat_xyzw_R = T_to_pos_quat_xyzw(T_R_G)

        T_R_O_prev = self.T_R_C @ T_C_O_prev
        object_position_R_prev, object_quat_xyzw_R_prev = T_to_pos_quat_xyzw(T_R_O_prev)

        T_R_O_prev_prev = self.T_R_C @ T_C_O_prev_prev
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
        if FABRIC_MODE == "PCA":
            num_hand_actions = 5
        elif FABRIC_MODE == "ALL":
            num_hand_actions = 16
        else:
            raise ValueError(f"Invalid FABRIC_MODE = {FABRIC_MODE}")

        hand_msg = Float64MultiArray()
        hand_msg.layout = MultiArrayLayout(
            dim=[
                MultiArrayDimension(
                    label="hand_target", size=num_hand_actions, stride=num_hand_actions
                )
            ],
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

                if FABRIC_MODE == "PCA":
                    num_hand_actions = 5
                elif FABRIC_MODE == "ALL":
                    num_hand_actions = 16
                else:
                    raise ValueError(f"Invalid FABRIC_MODE = {FABRIC_MODE}")
                assert_equals(hand_target.shape, (1, num_hand_actions))
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
                get_ros_loop_rate_str(
                    start_time=start_time,
                    before_sleep_time=before_sleep_time,
                    after_sleep_time=after_sleep_time,
                    node_name=rospy.get_name(),
                )
            )

    @property
    @functools.lru_cache()
    def T_R_C(self) -> np.ndarray:
        # Check camera parameter
        camera = rospy.get_param("/camera", None)
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
        rl_policy_node = RLPolicyNode()
        rl_policy_node.run()
    except rospy.ROSInterruptException:
        pass
