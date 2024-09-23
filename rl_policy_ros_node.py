#!/usr/bin/env python

from typing import Optional, Tuple

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout

from rl_player import RlPlayer


def var_to_is_none_str(var) -> str:
    if var is None:
        return "None"
    return "Not None"


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
        self.iiwa_joint_state_msg = None
        self.allegro_joint_state_msg = None
        self.fabric_state_msg = None

        # Subscribers
        self.object_pose_sub = rospy.Subscriber(
            "/object_pose", Pose, self.object_pose_callback
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
        self.num_observations = 138  # Update this number based on actual dimensions
        self.num_actions = 11  # First 6 for palm, last 5 for hand
        self.config_path = (
            "/juno/u/tylerlum/Downloads/config_resolved.yaml"  # Update this path
        )
        self.checkpoint_path = "/juno/u/tylerlum/Downloads/Pregrasp-LEFT_Track-POUR_move3_1gpu.pth"  # Update this path

        # Create the RL player
        self.player = RlPlayer(
            num_observations=self.num_observations,
            num_actions=self.num_actions,
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

        # Define limits for palm and hand targets
        self.palm_mins = torch.tensor([0, -0.4, 0.3, 0, 0, 0], device=self.device)
        self.palm_maxs = torch.tensor(
            [0.5, 0.4, 0.5, 2 * np.pi, 2 * np.pi, 2 * np.pi], device=self.device
        )
        self.hand_mins = torch.tensor(
            [0.2475, -0.3286, -0.7238, -0.0192, -0.5532], device=self.device
        )
        self.hand_maxs = torch.tensor(
            [3.8336, 3.0025, 0.8977, 1.0243, 0.0629], device=self.device
        )

    def object_pose_callback(self, msg: Pose):
        self.object_pose_msg = msg

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
            or self.fabric_state_msg is None
        ):
            rospy.logwarn(
                f"Waiting for all messages to be received... iiwa_joint_state_msg: {var_to_is_none_str(self.iiwa_joint_state_msg)}, allegro_joint_state_msg: {var_to_is_none_str(self.allegro_joint_state_msg)}, object_pose_msg: {var_to_is_none_str(self.object_pose_msg)}, fabric_state_msg: {var_to_is_none_str(self.fabric_state_msg)}"
            )
            return None

        # Concatenate the data from joint states and object pose
        iiwa_position = np.array(self.iiwa_joint_state_msg.position)
        iiwa_velocity = np.array(self.iiwa_joint_state_msg.velocity)

        allegro_position = np.array(self.allegro_joint_state_msg.position)
        allegro_velocity = np.array(self.allegro_joint_state_msg.velocity)

        object_position = np.array(
            [
                self.object_pose_msg.position.x,
                self.object_pose_msg.position.y,
                self.object_pose_msg.position.z,
            ]
        )
        object_orientation = np.array(
            [
                self.object_pose_msg.orientation.x,
                self.object_pose_msg.orientation.y,
                self.object_pose_msg.orientation.z,
                self.object_pose_msg.orientation.w,
            ]
        )

        fabric_q = np.array(self.fabric_state_msg.position)
        fabric_qd = np.array(self.fabric_state_msg.velocity)
        fabric_qdd = np.array(self.fabric_state_msg.effort)

        # Concatenate all observations into a 1D tensor
        observation = np.concatenate(
            [
                iiwa_position,
                iiwa_velocity,
                allegro_position,
                allegro_velocity,
                object_position,
                object_orientation,
                fabric_q,
                fabric_qd,
                fabric_qdd,
            ]
        )
        assert observation.shape == (self.num_observations,), f"{observation.shape}"

        return torch.from_numpy(observation).float().unsqueeze(0).to(self.device)

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
            # Create observation from the latest messages
            obs = self.create_observation()

            if obs is not None:
                # Get the normalized action from the RL player
                # normalized_action = self.player.get_normalized_action(obs=obs)
                normalized_action = torch.zeros(1, self.num_actions, device=self.device)
                assert normalized_action.shape == (
                    1,
                    self.num_actions,
                ), f"{normalized_action.shape}"

                # Rescale the action to get palm and hand targets
                palm_target, hand_target = self.rescale_action(normalized_action)

                # Publish the targets
                self.publish_targets(palm_target, hand_target)

            # Sleep to maintain 60Hz loop rate
            self.rate.sleep()


if __name__ == "__main__":
    try:
        rl_policy_node = RLPolicyNode()
        rl_policy_node.run()
    except rospy.ROSInterruptException:
        pass
