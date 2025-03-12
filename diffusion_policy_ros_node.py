#!/usr/bin/env python

import pybullet as p
import copy
import numpy as np
import rospy
import torch

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from std_msgs.msg import Header

from scipy.spatial.transform import Rotation as R

# Import your policy class
from dp_lowdim_policy import DiffusionUnetLowdimPolicy


def pose_msg_to_T(msg: Pose) -> np.ndarray:
    """
    Convert a geometry_msgs/Pose into a 4x4 homogeneous transform.
    """
    T = np.eye(4)
    T[:3, 3] = np.array([msg.position.x, msg.position.y, msg.position.z])
    rot_mat = R.from_quat([
        msg.orientation.x,
        msg.orientation.y,
        msg.orientation.z,
        msg.orientation.w
    ]).as_matrix()
    T[:3, :3] = rot_mat
    return T

def get_link_name_to_idx(robot: int) -> dict:
    link_name_to_idx = {}
    for i in range(p.getNumJoints(robot)):
        joint_info = p.getJointInfo(robot, i)
        link_name_to_idx[joint_info[12].decode("utf-8")] = i
    return link_name_to_idx

def set_robot_state(robot, q: np.ndarray) -> None:
    num_total_joints = p.getNumJoints(robot)
    actuatable_joint_idxs = [ i for i in range(num_total_joints) if p.getJointInfo(robot, i)[2] != p.JOINT_FIXED ]
    num_actuatable_joints = len(actuatable_joint_idxs)

    assert len(q.shape) == 1, f"q.shape: {q.shape}"
    assert (
        q.shape[0] <= num_actuatable_joints
    ), f"q.shape: {q.shape}, num_actuatable_joints: {num_actuatable_joints}"

    for i, joint_idx in enumerate(actuatable_joint_idxs):
        # q may not contain all the actuatable joints, so we assume that the joints not in q are all 0
        if i < len(q):
            p.resetJointState(robot, joint_idx, q[i])
        else:
            p.resetJointState(robot, joint_idx, 0)




class PolicyInferenceNode:
    def __init__(self):
        """
        - Initializes ROS node.
        - Loads the diffusion policy.
        - Subscribes/publishes to relevant topics.
        - Stores the latest messages in buffers.
        - Defines joint limits (first 7 are KUKA, last 16 are Allegro).
        """
        rospy.init_node("policy_inference_node")

        # ------------------------------
        # Set up inference rate
        # ------------------------------
        self.rate_hz = 15  # HARDCODED
        self.rate = rospy.Rate(self.rate_hz)

        # ------------------------------
        # Joint limits: first 7 are for iiwa, last 16 for allegro
        # ------------------------------
        self.joint_lower_limits = np.array([
            -2.96705972839, -2.09439510239, -2.96705972839,
            -2.09439510239, -2.96705972839, -2.09439510239,
            -3.05432619099,
            -0.47, -0.196, -0.174, -0.227,
            -0.47, -0.196, -0.174, -0.227,
            -0.47, -0.196, -0.174, -0.227,
             0.263, -0.105, -0.189, -0.162
        ])
        self.joint_upper_limits = np.array([
             2.96705972839,  2.09439510239,  2.96705972839,
             2.09439510239,  2.96705972839,  2.09439510239,
             3.05432619099,
             0.47, 1.61, 1.709, 1.618,
             0.47, 1.61, 1.709, 1.618,
             0.47, 1.61, 1.709, 1.618,
             1.396, 1.163, 1.644, 1.719
        ])
        assert len(self.joint_lower_limits) == 23, "Expected 23 total joints (7 + 16)."
        assert len(self.joint_upper_limits) == 23, "Expected 23 total joints (7 + 16)."

        # ------------------------------
        # Load the pybullet
        # ------------------------------
        p.connect(p.DIRECT)
        ROBOT_URDF = "/juno/u/oliviayl/repos/cross_embodiment/interactive_robot_visualizer/curobo/src/curobo/content/assets/robot/iiwa_allegro_description/kuka_allegro.urdf"
        self.robot_id = p.loadURDF(
                    ROBOT_URDF,
                    basePosition=[0, 0, 0],
                    baseOrientation=[0, 0, 0, 1],
                    useFixedBase=True,
                    flags=p.URDF_USE_INERTIA_FROM_FILE
                )
        self.robot_link_name_to_id = get_link_name_to_idx(self.robot_id)

        # ------------------------------
        # Load the diffusion policy
        # ------------------------------
        model_path = "/juno/u/oliviayl/repos/cross_embodiment/diffusion_policy/outputs/policies/snackbox_push_BC30_best_waypt.pth"
        self.policy = DiffusionUnetLowdimPolicy()
        model_state_dict = torch.load(model_path)["model_state_dict"]
        for k in model_state_dict.keys():
            if k.startswith("normalizer"):
                is_0_dim = len(model_state_dict[k].shape) == 0
                if is_0_dim:
                    model_state_dict[k] = model_state_dict[k].unsqueeze(0)
                else:
                    raise ValueError(f"Expected 0-dim normalizer, got {model_state_dict[k].shape}")
        self.policy.load_state_dict(model_state_dict)
        self.policy.eval()
        rospy.loginfo(f"Loaded policy from {model_path}")

        action_scale_path = "/juno/u/oliviayl/repos/cross_embodiment/diffusion_policy/outputs/action_scales/snackbox_push_action_norm_scale_waypt.npz"
        self.action_delta_scale = np.load(action_scale_path)["action_norm_scale"]
        assert self.action_delta_scale.shape == (23,), f"Expected action_delta_scale shape of (23,), got {self.action_delta_scale.shape}"

        # ------------------------------
        # Publishers
        # ------------------------------
        self.iiwa_cmd_pub = rospy.Publisher("/iiwa/joint_cmd", JointState, queue_size=10)
        self.allegro_cmd_pub = rospy.Publisher("/allegroHand_0/joint_cmd", JointState, queue_size=10)

        # ------------------------------
        # Subscribers (storing messages in buffers)
        # ------------------------------
        self.iiwa_joint_state_msg = None
        self.allegro_joint_state_msg = None
        self.object_pose_msg = None

        self.iiwa_sub = rospy.Subscriber("/iiwa/joint_states", JointState, self.iiwa_joint_state_callback)
        self.allegro_sub = rospy.Subscriber("/allegroHand_0/joint_states", JointState, self.allegro_joint_state_callback)
        self.object_pose_sub = rospy.Subscriber("/object_pose", Pose, self.object_pose_callback)

        rospy.loginfo("PolicyInferenceNode initialized.")

    # ------------------------------
    # ROS Callbacks: store latest messages
    # ------------------------------
    def iiwa_joint_state_callback(self, msg: JointState):
        self.iiwa_joint_state_msg = msg

    def allegro_joint_state_callback(self, msg: JointState):
        self.allegro_joint_state_msg = msg

    def object_pose_callback(self, msg: Pose):
        self.object_pose_msg = msg

    # ------------------------------
    # Main loop
    # ------------------------------
    def run(self):
        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            # Check that we have necessary messages
            if (self.iiwa_joint_state_msg is None or
                self.allegro_joint_state_msg is None or
                self.object_pose_msg is None):
                rospy.logwarn_throttle(5.0, "Waiting for all required messages...")
                self.rate.sleep()
                continue

            # Copy to avoid race conditions
            iiwa_joint_state_msg = copy.copy(self.iiwa_joint_state_msg)
            allegro_joint_state_msg = copy.copy(self.allegro_joint_state_msg)
            object_pose_msg = copy.copy(self.object_pose_msg)

            # ------------------------------
            # Build observation
            #  - 7 DoF for iiwa
            #  - 16 DoF for allegro
            #  - 16 for object pose (flatten 4x4)
            # => total 39 (example)
            # ------------------------------
            current_iiwa_q = np.array(iiwa_joint_state_msg.position)
            current_allegro_q = np.array(allegro_joint_state_msg.position)

            assert current_iiwa_q.shape == (7,), f"Expected 7 joints for iiwa, got {current_iiwa_q.shape}"
            assert current_allegro_q.shape == (16,), f"Expected 16 joints for allegro, got {current_allegro_q.shape}"
            current_q = np.concatenate([current_iiwa_q, current_allegro_q], axis=0)

            # Convert object_pose to 4x4
            T_C_O = pose_msg_to_T(object_pose_msg)

            # Hard-coded transform from camera to robot frame (example):
            T_R_C = np.eye(4)
            T_R_C[:3, :3] = np.array([
                [0.9543812680846684,  0.08746057618774912, -0.2854943830305726],
                [0.29537672607257903, -0.41644924520026877,  0.8598387150313551],
                [-0.043691930876822334, -0.904942359371598, -0.42328517738189414]
            ])
            T_R_C[:3, 3] = np.array([0.5947949577333569, -0.9635715691360609, 0.6851893282998003])

            # Transform object pose from camera frame to robot frame
            T_R_O = T_R_C @ T_C_O
            assert T_R_O.shape == (4, 4), f"T_R_O shape mismatch: {T_R_O.shape}"

            # Flatten the 4x4
            flat_object_pose = T_R_O.reshape(16)

            set_robot_state(self.robot_id, current_q)
            robot_palm_com, robot_palm_quat, *_ = p.getLinkState(
                self.robot_id,
                self.robot_link_name_to_id["palm_link"],
                computeForwardKinematics=1,
            )
            # Combine (23 + 16 + 7 = 46) for obs
            curr_obs = np.concatenate((current_q, robot_palm_com, robot_palm_quat, flat_object_pose), axis=0)
            assert curr_obs.shape[0] == 46, f"curr_obs.shape: {curr_obs.shape}"

            obs_torch = torch.from_numpy(curr_obs).float().unsqueeze(0)  # (1, 46)

            # ------------------------------
            # Policy inference
            #  "predict_action" returns a dict, e.g. {"action": <torch.Tensor>}
            # ------------------------------
            with torch.no_grad():
                raw_action_dict = self.policy.predict_action({"obs": obs_torch})
            raw_action_delta = raw_action_dict["action"].cpu().numpy().squeeze(0)
            HORIZON_LEN = 8
            assert raw_action_delta.shape == (HORIZON_LEN, 23), f"Expected ({HORIZON_LEN}, 23)-dim action, got {raw_action_delta.shape}"
            raw_action_delta = raw_action_delta[0]
            scaled_action_delta = raw_action_delta * self.action_delta_scale

            # ------------------------------
            # Combine current robot state => full 23
            # first 7 are iiwa, next 16 are allegro
            # ------------------------------
            current_q = np.concatenate([current_iiwa_q, current_allegro_q], axis=0)
            assert current_q.shape == (23,)

            new_q = current_q + scaled_action_delta  # delta from current position

            # Clip to joint limits
            new_q = np.clip(new_q, self.joint_lower_limits, self.joint_upper_limits)

            # Split again
            new_iiwa_q = new_q[:7]
            new_allegro_q = new_q[7:]

            # ------------------------------
            # Publish commands
            # ------------------------------
            current_time = rospy.Time.now()

            # KUKA
            iiwa_cmd_msg = JointState()
            iiwa_cmd_msg.header = Header(stamp=current_time)
            iiwa_cmd_msg.name = [f"iiwa_joint_{i+1}" for i in range(7)]
            iiwa_cmd_msg.position = new_iiwa_q.tolist()
            self.iiwa_cmd_pub.publish(iiwa_cmd_msg)

            # Allegro
            allegro_cmd_msg = JointState()
            allegro_cmd_msg.header = Header(stamp=current_time)
            allegro_cmd_msg.name = [f"allegro_joint_{i}" for i in range(16)]
            allegro_cmd_msg.position = new_allegro_q.tolist()
            self.allegro_cmd_pub.publish(allegro_cmd_msg)

            # ------------------------------
            # Sleep to maintain rate
            # ------------------------------
            before_sleep_time = rospy.Time.now()
            self.rate.sleep()
            after_sleep_time = rospy.Time.now()

            total_loop_time = (after_sleep_time - start_time).to_sec()
            rospy.loginfo_throttle(
                2.0,
                f"[{rospy.get_name()}] Loop took {total_loop_time:.4f}s "
                f"(~{1.0/total_loop_time:.2f} Hz actual)."
            )


def main():
    node = PolicyInferenceNode()
    node.run()

if __name__ == '__main__':
    main()
